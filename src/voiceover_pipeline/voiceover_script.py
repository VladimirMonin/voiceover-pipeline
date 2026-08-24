import re
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_ELEVENLABS_VOICE,
    DEFAULT_FALLBACK_VOICE,
    DEFAULT_OMNIVOICE_VOICE,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_OPENROUTER_TTS_VOICE,
    DEFAULT_POLZA_TTS_VOICE,
    DEFAULT_PROVIDER,
    DEFAULT_QWEN_VOICE,
    DEFAULT_VOICE,
    ELEVENLABS_TTS_VOICES,
    GEMINI_TTS_VOICES,
    OMNIVOICE_LOCAL_MODEL_ID,
    OPENAI_TTS_VOICES,
    OPENROUTER_TTS_MODELS,
    POLZA_TTS_MODELS,
    PROVIDER_DEFAULT_MODELS,
    QWEN_PRESET_SPEAKERS,
)
from .gemini_dialogue import error, parse_frontmatter, warning
from .local_tts_text import get_local_tts_chunk_profile
from .models import ScriptChunk

VOICEOVER_FORMAT = "voiceover"
PROMPT_SKELETON_MARKERS = [
    "Synthesize speech",
    "AUDIO PROFILE",
    "SCENE",
    "PERFORMANCE",
    "CONTEXT",
    "#### TRANSCRIPT",
]
SUPPORTED_PROVIDERS = [
    "polza-chat-audio",
    "polza-tts",
    "openrouter-tts",
    "qwen-local",
    "omnivoice-local",
]
POLZA_CHAT_AUDIO_MODELS = ["openai/gpt-audio-mini", "openai/gpt-audio"]
POLZA_CHAT_AUDIO_VOICES = [
    "ash",
    "ballad",
    "coral",
    "verse",
    "marin",
    "cedar",
    "echo",
    "sage",
    "shimmer",
    "onyx",
]
MODELS_BY_PROVIDER = {
    "polza-chat-audio": POLZA_CHAT_AUDIO_MODELS,
    "polza-tts": POLZA_TTS_MODELS,
    "openrouter-tts": OPENROUTER_TTS_MODELS,
    "omnivoice-local": [OMNIVOICE_LOCAL_MODEL_ID],
}
DEFAULT_MAX_CHUNK_CHARS = 2000


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
    max_chunk_chars: int | None = None,
    agent: bool = False,
) -> dict[str, Any]:
    text = script_path.read_text(encoding="utf-8-sig")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    meta, body, body_start_line, fm_errors = parse_frontmatter(text)
    errors.extend(fm_errors)

    if meta.get("format") != VOICEOVER_FORMAT:
        errors.append(
            error("FORMAT_NOT_VOICEOVER", "Frontmatter must contain format: voiceover.", line=1)
        )

    provider = provider_override or str(meta.get("provider") or meta.get("service") or "")
    if not provider:
        errors.append(
            error("PROVIDER_MISSING", "Frontmatter must define provider or service.", line=1)
        )
        provider = DEFAULT_PROVIDER
    if provider not in SUPPORTED_PROVIDERS:
        errors.append(
            error(
                "INVALID_PROVIDER",
                f"Unsupported provider: {provider}.",
                actual=provider,
                expected=SUPPORTED_PROVIDERS,
            )
        )

    model = model_override or str(meta.get("model") or PROVIDER_DEFAULT_MODELS.get(provider, ""))
    if not model:
        errors.append(
            error("MODEL_MISSING", "Model is required for this provider.", line=1, actual=model)
        )
    validate_model(provider, model, errors)

    voice = voice_override or str(meta.get("voice") or default_voice(provider, model))
    validate_voice(provider, model, voice, errors)

    profile = get_local_tts_chunk_profile(provider, model)
    resolved_max_chunk_chars = (
        profile.target_chars
        if profile is not None and max_chunk_chars is None
        else max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS
    )
    raw_digit_policy = "reject" if profile is not None and profile.reject_raw_digits else "warn"

    fallback_voice = str(meta.get("fallback_voice") or DEFAULT_FALLBACK_VOICE)
    if provider == "polza-chat-audio" and fallback_voice not in POLZA_CHAT_AUDIO_VOICES:
        errors.append(
            error(
                "INVALID_FALLBACK_VOICE",
                f"Fallback voice {fallback_voice} is invalid.",
                actual=fallback_voice,
                expected=POLZA_CHAT_AUDIO_VOICES,
            )
        )
    if provider != "polza-chat-audio" and meta.get("fallback_voice"):
        warnings.append(
            warning(
                "FALLBACK_VOICE_IGNORED",
                "fallback_voice is only used by polza-chat-audio.",
                actual=fallback_voice,
            )
        )

    style_prompt = str(meta.get("style_prompt") or meta.get("prompt") or "")
    if style_prompt and not supports_style_prompt(provider, model):
        warnings.append(
            warning(
                "STYLE_PROMPT_IGNORED",
                f"style_prompt is ignored by provider/model {provider}/{model}.",
                actual=style_prompt[:80],
            )
        )

    chunks = split_body_chunks(body, body_start_line, delimiter)
    if not chunks:
        errors.append(
            error("CHUNK_EMPTY", "Script body contains no non-empty chunks.", line=body_start_line)
        )

    chunk_reports: list[dict[str, Any]] = []
    total_chars = 0
    for chunk in chunks:
        report = validate_plain_chunk(chunk, resolved_max_chunk_chars, raw_digit_policy, agent)
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
        "max_chunk_chars": resolved_max_chunk_chars,
        "spoken_text": aggregate_spoken_text(chunk_reports, raw_digit_policy),
        "chunk_reports": chunk_reports,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "max_chunk_chars": resolved_max_chunk_chars,
            "spoken_text": aggregate_spoken_text(chunk_reports, raw_digit_policy),
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
        errors.append(
            error(
                "INVALID_MODEL",
                f"Model {model} is invalid for provider {provider}.",
                actual=model,
                expected=valid,
            )
        )


def validate_voice(provider: str, model: str, voice: str, errors: list[dict[str, Any]]) -> None:
    voices = voices_for_provider_model(provider, model)
    if voices and voice not in voices:
        errors.append(
            error(
                "INVALID_VOICE",
                f"Voice {voice} is invalid for provider/model {provider}/{model}.",
                actual=voice,
                expected=voices,
            )
        )


def voices_for_provider_model(provider: str, model: str) -> list[str]:
    if provider == "polza-chat-audio":
        return POLZA_CHAT_AUDIO_VOICES
    if provider == "polza-tts":
        return ELEVENLABS_TTS_VOICES if model.startswith("elevenlabs/") else OPENAI_TTS_VOICES
    if provider == "openrouter-tts":
        return OPENAI_TTS_VOICES if model.startswith("openai/") else GEMINI_TTS_VOICES
    if provider == "qwen-local":
        return QWEN_PRESET_SPEAKERS
    if provider == "omnivoice-local":
        return [DEFAULT_OMNIVOICE_VOICE]
    return []


def default_voice(provider: str, model: str) -> str:
    if provider == "polza-tts":
        return (
            DEFAULT_ELEVENLABS_VOICE if model.startswith("elevenlabs/") else DEFAULT_POLZA_TTS_VOICE
        )
    if provider == "openrouter-tts":
        return (
            DEFAULT_OPENAI_TTS_VOICE
            if model.startswith("openai/")
            else DEFAULT_OPENROUTER_TTS_VOICE
        )
    if provider == "qwen-local":
        return DEFAULT_QWEN_VOICE
    if provider == "omnivoice-local":
        return DEFAULT_OMNIVOICE_VOICE
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


def append_chunk(
    chunks: list[dict[str, Any]], lines: list[tuple[int, str]], fallback_line: int
) -> None:
    text = "\n".join(text for _, text in lines).strip()
    if not text:
        return
    nonempty = [(line_no, text) for line_no, text in lines if text.strip()]
    chunks.append(
        {
            "chunk": len(chunks) + 1,
            "line_start": nonempty[0][0] if nonempty else fallback_line,
            "line_end": nonempty[-1][0] if nonempty else fallback_line,
            "text": text,
        }
    )


def validate_plain_chunk(
    chunk: dict[str, Any], max_chunk_chars: int, raw_digit_policy: str, agent: bool
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    text = chunk["text"]
    chars = len(text)
    if chars > max_chunk_chars:
        errors.append(
            error(
                "CHUNK_TOO_LARGE",
                f"Chunk {chunk['chunk']} is {chars} chars, limit is {max_chunk_chars}.",
                chunk=chunk["chunk"],
                line_start=chunk["line_start"],
                line_end=chunk["line_end"],
                actual=chars,
                limit=max_chunk_chars,
                snippet=text[:160] if agent else None,
                suggested_fix=f"Split chunk {chunk['chunk']} before line {(chunk['line_start'] + chunk['line_end']) // 2}.",
            )
        )
    if any(ch.isdigit() for ch in text):
        issue = (
            error(
                "RAW_DIGITS",
                "Chunk contains raw digits; write numbers, dates, percentages, fractions, and versions in words.",
                chunk=chunk["chunk"],
            )
            if raw_digit_policy == "reject"
            else warning(
                "CONTAINS_DIGITS",
                "Chunk contains digits; TTS pronunciation may be unexpected.",
                chunk=chunk["chunk"],
            )
        )
        (errors if raw_digit_policy == "reject" else warnings).append(issue)
    leaked_markers = [marker for marker in PROMPT_SKELETON_MARKERS if marker in text]
    if leaked_markers:
        warnings.append(
            warning(
                "PROMPT_SKELETON_IN_BODY",
                "Prompt direction markers found in the spoken body. Move direction into frontmatter style_prompt/prompt and keep body as transcript only.",
                chunk=chunk["chunk"],
                actual=leaked_markers,
            )
        )
    return {
        "chunk": chunk["chunk"],
        "line_start": chunk["line_start"],
        "line_end": chunk["line_end"],
        "chars": chars,
        "text": text,
        "spoken_text": spoken_text_metrics(text, raw_digit_policy),
        "errors": errors,
        "warnings": warnings,
    }


def spoken_text_metrics(text: str, raw_digit_policy: str) -> dict[str, Any]:
    """Return deterministic language composition metrics for spoken text."""
    letters = [char for char in text if char.isalpha()]
    latin_alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    latin_characters = sum(char in latin_alphabet for char in letters)
    latin_words = 0
    mixed_script_words = 0
    for token in re.findall(r"\S+", text):
        token_letters = [char for char in token if char.isalpha()]
        if not token_letters:
            continue
        has_latin = any(char in latin_alphabet for char in token_letters)
        has_non_latin = any(char not in latin_alphabet for char in token_letters)
        if has_latin and has_non_latin:
            mixed_script_words += 1
        elif has_latin:
            latin_words += 1
    total_letters = len(letters)
    return {
        "raw_digit_policy": raw_digit_policy,
        "latin_characters": latin_characters,
        "latin_words": latin_words,
        "mixed_script_words": mixed_script_words,
        "total_letters": total_letters,
        "latin_ratio": latin_characters / total_letters if total_letters else 0.0,
    }


def aggregate_spoken_text(
    chunk_reports: list[dict[str, Any]], raw_digit_policy: str
) -> dict[str, Any]:
    """Aggregate spoken-text metrics while preserving the policy field."""
    metrics = [report["spoken_text"] for report in chunk_reports]
    total_letters = sum(item["total_letters"] for item in metrics)
    latin_characters = sum(item["latin_characters"] for item in metrics)
    return {
        "raw_digit_policy": raw_digit_policy,
        "latin_characters": latin_characters,
        "latin_words": sum(item["latin_words"] for item in metrics),
        "mixed_script_words": sum(item["mixed_script_words"] for item in metrics),
        "total_letters": total_letters,
        "latin_ratio": latin_characters / total_letters if total_letters else 0.0,
    }
