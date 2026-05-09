from pathlib import Path

from voiceover_pipeline.config import (
    POLZA_PROMPTABLE_TTS_MODELS,
    PROMPTABLE_TTS_MODELS,
    TTS_PROMPT_MODE_NATIVE,
    TTS_PROMPT_MODE_NONE,
    TTS_PROMPT_MODE_PREFIX,
)


def resolve_prompt_mode(provider_id: str, model: str, prompt_mode: str | None = None) -> str:
    if prompt_mode == TTS_PROMPT_MODE_NONE:
        return TTS_PROMPT_MODE_NONE
    if prompt_mode in (TTS_PROMPT_MODE_PREFIX, TTS_PROMPT_MODE_NATIVE):
        return prompt_mode

    if provider_id == "openrouter-tts" and model in PROMPTABLE_TTS_MODELS:
        return PROMPTABLE_TTS_MODELS[model]
    if provider_id == "polza-tts" and model in POLZA_PROMPTABLE_TTS_MODELS:
        return POLZA_PROMPTABLE_TTS_MODELS[model]

    if model.startswith("google/"):
        return TTS_PROMPT_MODE_NATIVE
    if model.startswith("openai/"):
        return TTS_PROMPT_MODE_NONE

    return TTS_PROMPT_MODE_NONE


def build_prompted_input(text: str, style_prompt: str | None, prompt_mode: str) -> str:
    if prompt_mode == TTS_PROMPT_MODE_NONE or style_prompt is None:
        return text
    if prompt_mode == TTS_PROMPT_MODE_PREFIX:
        return f"{style_prompt}\n\n{text}"
    return text


def build_request_body(
    model: str,
    text: str,
    voice: str,
    response_format: str,
    style_prompt: str | None,
    prompt_mode: str,
) -> dict:
    body: dict = {
        "model": model,
        "voice": voice,
        "response_format": response_format,
    }

    if prompt_mode == TTS_PROMPT_MODE_NATIVE and style_prompt is not None:
        body["input"] = text
        body["prompt"] = style_prompt
    elif prompt_mode == TTS_PROMPT_MODE_PREFIX:
        body["input"] = build_prompted_input(text, style_prompt, TTS_PROMPT_MODE_PREFIX)
    else:
        body["input"] = text

    return body


def read_style_prompt_from_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Style prompt file not found: {file_path}")
    content = file_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError(f"Style prompt file is empty: {file_path}")
    return content
