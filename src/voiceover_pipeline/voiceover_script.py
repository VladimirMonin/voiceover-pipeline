from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_FALLBACK_VOICE,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_OPENROUTER_TTS_VOICE,
    DEFAULT_POLZA_TTS_VOICE,
    DEFAULT_PROVIDER,
    DEFAULT_QWEN_VOICE,
    DEFAULT_VOICE,
    ELEVENLABS_TTS_VOICES,
    GEMINI_TTS_VOICES,
    OPENAI_TTS_VOICES,
    OPENROUTER_TTS_MODELS,
    POLZA_TTS_MODELS,
    PROVIDER_DEFAULT_MODELS,
    QWEN_PRESET_SPEAKERS,
)
from .gemini_dialogue import error, parse_frontmatter, warning
from .models import ScriptChunk


VOICEOVER_FORMAT = "voiceover"
SUPPORTED_PROVIDERS = ["polza-chat-audio", "polza-tts", "openrouter-tts", "qwen-local"]
POLZA_CHAT_AUDIO_MODELS = ["openai/gpt-audio-mini", "openai/gpt-audio"]
POLZA_CHAT_AUDIO_VOICES = ["ash", "ballad", "coral", "verse", "marin", "cedar", "echo", "sage", "shimmer", "onyx"]
MODELS_BY_PROVIDER = {
    "polza-chat-audio": POLZA_CHAT_AUDIO_MODELS,
    "polza-tts": POLZA_TTS_MODELS,
    "openrouter-tts": OPENROUTER_TTS_MODELS,
}


def detect_frontmatter_format(script_path: Path) -> str | None:
    try:
        text = script_path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    meta, _, _, errors = parse_frontmatter(text)
    if errors:
        return None
    raw_format = meta.get("format")
    return str(raw_format) if raw_format else None


def validate_voiceover_file(
    script_path: Path,
    delimiter: str = "******",
    provider_override: str | None = None,
    model_override: str | None = None,
    voice_override: str | None = None,
    max_chunk_chars: int = 2000,
    agent: bool = False,
) -> dict[str, Any]:
    text = script_path.read_text(encoding="utf-8-sig")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    meta, body, body_start_line, fm_errors = parse_frontmatter(text)
    errors.extend(fm_errors)

    if meta.get("format") != VOICEOVER_FORMAT:
        errors.append(error("FORMAT_NOT_VOICEOVER", "Frontmatter must contain format: voiceover.", line=1))

    provider = provider_override or str(meta.get("provider") or meta.get("service") or "")
    if not provider:
        errors.append(error("PROVIDER_MISSING", "Frontmatter must define provider or service.", line=1))
        provider = DEFAULT_PROVIDER
    if provider not in SUPPORTED_PROVIDERS:
        errors.append(error("INVALID_PROVIDER", f"Unsupported provider: {provider}.", actual=provider, expected=SUPPORTED_PROVIDERS))

    model = model_override or str(meta.get("model") or PROVIDER_DEFAULT_MODELS.get(provider, ""))
    if not model:
        errors.append(error("MODEL_MISSING", "Model is required for this provider.", line=1, actual=model))
    validate_model(provider, model, errors)

    voice = voice_override or str(meta.get("voice") or default_voice(provider, model))
    validate_voice(provider, model, voice, errors)

    fallback_voice = str(meta.get("fallback_voice") or DEFAULT_FALLBACK_VOICE)
    if provider == "polza-chat-audio" and fallback_voice not in POLZA_CHAT_AUDIO_VOICES:
        errors.append(error("INVALID_FALLBACK_VOICE", f"Fallback voice {fallback_voice} is invalid.", actual=fallback_voice, expected=POLZA_CHAT_AUDIO_VOICES))
    if provider != "polza-chat-audio" and meta.get("fallback_voice"):
        warnings.append(warning("FALLBACK_VOICE_IGNORED", "fallback_voice is only used by polza-chat-audio.", actual=fallback_voice))

    style_prompt = str(meta.get("style_prompt") or meta.get("prompt") or "")
    if style_prompt and not supports_style_prompt(provider, model):
        warnings.append(warning("STYLE_PROMPT_IGNORED", f"style_prompt is ignored by provider/model {provider}/{model}.", actual=style_prompt[:80]))

    chunks = split_body_chunks(body, body_start_line, delimiter)
    if not chunks:
        errors.append(error("CHUNK_EMPTY", "Script body contains no non-empty chunks.", line=body_start_line))

    chunk_reports: list[dict[str, Any]] = []
    total_chars = 0
    for chunk in chunks:
        report = validate_plain_chunk(chunk, max_chunk_chars, agent)
        errors.extend(report.pop("errors"))
        warnings.extend(report.pop("warnings"))
        total_chars += report["chars"]
        chunk_reports.append(report)

    valid = not errors
    return {
        "status": "success" if valid else "error",
        "valid": valid,
        "format": VOICEOVER_FORMAT,
        "script": str(script_path),
        "effective_config": {
            "provider": provider,
            "model": model,
            "voice": voice,
            "fallback_voice": fallback_voice if provider == "polza-chat-audio" else None,
            "style_prompt": style_prompt or None,
        },
        "chunks": len(chunks),
        "total_chars": total_chars,
        "max_chunk_chars": max_chunk_chars,
        "chunk_reports": chunk_reports,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "max_chunk_chars": max_chunk_chars,
        },
    }


def chunks_from_voiceover_report(report: dict[str, Any]) -> list[ScriptChunk]:
    return [
        ScriptChunk(number=item["chunk"], id=f"chunk_{item['chunk']:02d}", text=item["text"])
        for item in report.get("chunk_reports", [])
    ]


def validate_model(provider: str, model: str, errors: list[dict[str, Any]]) -> None:
    valid = MODELS_BY_PROVIDER.get(provider)
    if valid and model not in valid:
        errors.append(error("INVALID_MODEL", f"Model {model} is invalid for provider {provider}.", actual=model, expected=valid))


def validate_voice(provider: str, model: str, voice: str, errors: list[dict[str, Any]]) -> None:
    voices = voices_for_provider_model(provider, model)
    if voices and voice not in voices:
        errors.append(error("INVALID_VOICE", f"Voice {voice} is invalid for provider/model {provider}/{model}.", actual=voice, expected=voices))


def voices_for_provider_model(provider: str, model: str) -> list[str]:
    if provider == "polza-chat-audio":
        return POLZA_CHAT_AUDIO_VOICES
    if provider == "polza-tts":
        return ELEVENLABS_TTS_VOICES if model.startswith("elevenlabs/") else OPENAI_TTS_VOICES
    if provider == "openrouter-tts":
        return OPENAI_TTS_VOICES if model.startswith("openai/") else GEMINI_TTS_VOICES
    if provider == "qwen-local":
        return QWEN_PRESET_SPEAKERS
    return []


def default_voice(provider: str, model: str) -> str:
    if provider == "polza-tts":
        return DEFAULT_ELEVENLABS_VOICE if model.startswith("elevenlabs/") else DEFAULT_POLZA_TTS_VOICE
    if provider == "openrouter-tts":
        return DEFAULT_OPENAI_TTS_VOICE if model.startswith("openai/") else DEFAULT_OPENROUTER_TTS_VOICE
    if provider == "qwen-local":
        return DEFAULT_QWEN_VOICE
    return DEFAULT_VOICE


def supports_style_prompt(provider: str, model: str) -> bool:
    return provider == "openrouter-tts" and model.startswith("google/")


def split_body_chunks(body: str, body_start_line: int, delimiter: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[tuple[int, str]] = []
    for offset, line in enumerate(body.splitlines(), start=body_start_line):
        if line.strip() == delimiter:
            append_chunk(chunks, current, offset)
            current = []
        else:
            current.append((offset, line))
    append_chunk(chunks, current, body_start_line + len(body.splitlines()))
    return chunks


def append_chunk(chunks: list[dict[str, Any]], lines: list[tuple[int, str]], fallback_line: int) -> None:
    text = "\n".join(text for _, text in lines).strip()
    if not text:
        return
    nonempty = [(line_no, text) for line_no, text in lines if text.strip()]
    chunks.append({
        "chunk": len(chunks) + 1,
        "line_start": nonempty[0][0] if nonempty else fallback_line,
        "line_end": nonempty[-1][0] if nonempty else fallback_line,
        "text": text,
    })


def validate_plain_chunk(chunk: dict[str, Any], max_chunk_chars: int, agent: bool) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    text = chunk["text"]
    chars = len(text)
    if chars > max_chunk_chars:
        errors.append(error(
            "CHUNK_TOO_LARGE",
            f"Chunk {chunk['chunk']} is {chars} chars, limit is {max_chunk_chars}.",
            chunk=chunk["chunk"],
            line_start=chunk["line_start"],
            line_end=chunk["line_end"],
            actual=chars,
            limit=max_chunk_chars,
            snippet=text[:160] if agent else None,
            suggested_fix=f"Split chunk {chunk['chunk']} before line {(chunk['line_start'] + chunk['line_end']) // 2}.",
        ))
    if any(ch.isdigit() for ch in text):
        warnings.append(warning("CONTAINS_DIGITS", "Chunk contains digits; TTS pronunciation may be unexpected.", chunk=chunk["chunk"]))
    return {
        "chunk": chunk["chunk"],
        "line_start": chunk["line_start"],
        "line_end": chunk["line_end"],
        "chars": chars,
        "text": text,
        "errors": errors,
        "warnings": warnings,
    }
