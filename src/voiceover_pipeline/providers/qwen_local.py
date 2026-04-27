from __future__ import annotations

from pathlib import Path
from typing import Any

from voiceover_pipeline.config import (
    QWEN_INSTRUCT,
    QWEN_LANGUAGE,
    QWEN_MODEL_BASE,
    QWEN_MODEL_CUSTOMVOICE,
)
from voiceover_pipeline.models import SynthesisResult
from voiceover_pipeline.providers.base import TTSProvider


class QwenLocalTTSProvider(TTSProvider):
    provider_id = "qwen-local"

    def __init__(
        self,
        mode: str = "preset",
        voice: str | None = None,
        instruct: str = QWEN_INSTRUCT,
        language: str = QWEN_LANGUAGE,
        sample_path: str | None = None,
        sample_text: str = "",
        temp_dir: str = "temp",
    ) -> None:
        self._mode = mode
        self._voice = voice
        self._instruct = instruct
        self._language = language
        self._sample_path = sample_path
        self._sample_text = sample_text
        self._temp_dir = Path(temp_dir)

        self._model: Any = None

    def _load_model(self) -> Any:
        if self._mode == "preset":
            model_name = QWEN_MODEL_CUSTOMVOICE
        elif self._mode == "clone":
            model_name = QWEN_MODEL_BASE
        else:
            raise ValueError(f"Unknown qwen mode: {self._mode}")

        from qwen_tts import Qwen3TTSModel
        import torch

        from voiceover_pipeline.config import QWEN_ATTN_IMPL, QWEN_DEVICE

        device = QWEN_DEVICE if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        return Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation=QWEN_ATTN_IMPL,
        )

    def synthesize_chunk(self, text: str, chunk_id: str) -> SynthesisResult:
        if self._model is None:
            print(f"Loading Qwen3-TTS model ({self._mode}) ...")
            self._model = self._load_model()
            print("Model loaded.")

        self._temp_dir.mkdir(parents=True, exist_ok=True)

        if self._mode == "preset":
            voice = self._voice or "Aiden"
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=self._language,
                speaker=voice,
                instruct=self._instruct,
            )
        elif self._mode == "clone":
            ref_audio = self._sample_path
            if not ref_audio or not Path(ref_audio).exists():
                raise FileNotFoundError(
                    f"Reference audio not found for clone mode: {ref_audio}"
                )

            xvec_only = not bool(self._sample_text)
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=self._language,
                ref_audio=ref_audio,
                ref_text=self._sample_text or None,
                x_vector_only_mode=xvec_only,
            )
        else:
            raise ValueError(f"Unknown qwen mode: {self._mode}")

        wav_path = self._temp_dir / f"{chunk_id}.wav"
        import soundfile as sf
        sf.write(str(wav_path), wavs[0], sr)
        wav_bytes = wav_path.read_bytes()

        return SynthesisResult(
            audio_bytes=wav_bytes,
            audio_format="wav",
            transcript=text,
            client_path="qwen-local",
            raw_metadata={
                "voice": voice if self._mode == "preset" else "clone",
                "provider": self.provider_id,
                "mode": self._mode,
            },
        )
