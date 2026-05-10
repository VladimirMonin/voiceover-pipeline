import re
from pathlib import Path
from typing import Any

from .config import GEMINI_TTS_VOICES
from .models import ScriptChunk


GEMINI_DIALOGUE_FORMAT = "gemini-dialogue"
GEMINI_TTS_MODEL = "google/gemini-3.1-flash-tts-preview"
DEFAULT_MAX_CHUNK_BYTES = 3500
HARD_MAX_CHUNK_BYTES = 4000
DEFAULT_ALLOWED_TAGS = {
    "amazed",
    "calmly",
    "confidently",
    "cough",
    "crying",
    "curious",
    "excited",
    "excitedly",
    "gasp",
    "gently",
    "giggles",
    "laughs",
    "long pause",
    "medium pause",
    "mischievously",
    "panicked",
    "sadly",
    "sarcastic",
    "serious",
    "shouting",
    "short pause",
    "sighs",
    "thoughtfully",
    "tired",
    "trembling",
    "uhm",
    "warmly",
    "whispers",
}
PROMPT_SKELETON_MARKERS = [
    "Synthesize speech",
    "AUDIO PROFILE",
    "SCENE",
    "PERFORMANCE",
    "CONTEXT",
    "#### TRANSCRIPT",
]

_SPEAKER_RE = re.compile(r"^([A-Za-z0-9]+):\s*(.*)$")
_TAG_RE = re.compile(r"\[([^\[\]\n]+)\]")


def validate_gemini_dialogue_file(
    script_path: Path,
    delimiter: str = "******",
    model: str | None = None,
    speaker_voice_overrides: list[str] | None = None,
    agent: bool = False,
) -> dict[str, Any]:
    text = script_path.read_text(encoding="utf-8-sig")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    meta, body, body_start_line, fm_errors = parse_frontmatter(text)
    errors.extend(fm_errors)

    if meta.get("format") != GEMINI_DIALOGUE_FORMAT:
        errors.append(error("FORMAT_NOT_GEMINI_DIALOGUE", "Frontmatter must contain format: gemini-dialogue.", line=1))

    active_model = model or str(meta.get("model") or "")
    if active_model != GEMINI_TTS_MODEL:
        errors.append(error(
            "MODEL_NOT_GEMINI_TTS",
            f"Gemini dialogue requires model {GEMINI_TTS_MODEL}.",
            line=1,
            actual=active_model or None,
            expected=GEMINI_TTS_MODEL,
        ))

    speaker_voice_map = extract_speaker_voice_map(meta, errors)
    apply_speaker_voice_overrides(speaker_voice_map, speaker_voice_overrides or [], errors)
    validate_speakers(speaker_voice_map, errors)

    allowed_tags = extract_allowed_tags(meta)
    max_chunk_bytes = extract_max_chunk_bytes(meta, warnings)
    chunks = split_dialogue_chunks(body, body_start_line, delimiter)
    if not chunks:
        errors.append(error("CHUNK_EMPTY", "Script body contains no non-empty chunks.", line=body_start_line))

    chunk_reports: list[dict[str, Any]] = []
    total_bytes = 0
    for chunk in chunks:
        report = validate_chunk(chunk, speaker_voice_map, allowed_tags, max_chunk_bytes, agent)
        errors.extend(report.pop("errors"))
        warnings.extend(report.pop("warnings"))
        chunk_reports.append(report)
        total_bytes += report["utf8_bytes"]

    style_prompt = build_style_prompt(meta, speaker_voice_map)
    valid = not errors
    return {
        "status": "success" if valid else "error",
        "valid": valid,
        "format": GEMINI_DIALOGUE_FORMAT,
        "model": active_model,
        "script": str(script_path),
        "chunks": len(chunks),
        "total_utf8_bytes": total_bytes,
        "speaker_voice_map": speaker_voice_map,
        "style_prompt": style_prompt,
        "max_chunk_bytes": max_chunk_bytes,
        "hard_max_chunk_bytes": HARD_MAX_CHUNK_BYTES,
        "chunk_reports": chunk_reports,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "max_chunk_bytes": max_chunk_bytes,
        },
    }


def chunks_from_validation(report: dict[str, Any]) -> list[ScriptChunk]:
    return [
        ScriptChunk(number=item["chunk"], id=f"chunk_{item['chunk']:02d}", text=item["text"])
        for item in report.get("chunk_reports", [])
    ]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str, int, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, 1, [error("FRONTMATTER_MISSING", "Gemini dialogue script requires YAML-like frontmatter.", line=1)]

    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        return {}, "", len(lines) + 1, [error("FRONTMATTER_YAML_INVALID", "Frontmatter closing --- was not found.", line=1)]

    try:
        meta = parse_simple_yaml(lines[1:end_index], start_line=2)
    except ValueError as exc:
        meta = {}
        errors.append(error("FRONTMATTER_YAML_INVALID", str(exc), line=1))

    body = "\n".join(lines[end_index + 1:])
    return meta, body, end_index + 2, errors


def parse_simple_yaml(lines: list[str], start_line: int = 1) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            idx += 1
            continue
        if raw.startswith(" "):
            raise ValueError(f"Unexpected indented line at {start_line + idx}: {line}")
        if ":" not in line:
            raise ValueError(f"Expected key: value at {start_line + idx}: {line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == ">":
            block: list[str] = []
            idx += 1
            while idx < len(lines) and (lines[idx].startswith(" ") or not lines[idx].strip()):
                block.append(lines[idx].strip())
                idx += 1
            meta[key] = " ".join(part for part in block if part).strip()
            continue
        if not value:
            if key == "speakers":
                speakers, idx = parse_speakers(lines, idx + 1, start_line)
                meta[key] = speakers
                continue
            items, next_idx = parse_list(lines, idx + 1)
            if items:
                meta[key] = items
                idx = next_idx
                continue
            meta[key] = {}
            idx += 1
            continue
        meta[key] = parse_scalar(value)
        idx += 1
    return meta


def parse_speakers(lines: list[str], idx: int, start_line: int) -> tuple[dict[str, Any], int]:
    speakers: dict[str, Any] = {}
    current: str | None = None
    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip():
            idx += 1
            continue
        if not raw.startswith("  "):
            break
        if raw.startswith("  ") and not raw.startswith("    "):
            line = raw.strip()
            if not line.endswith(":"):
                raise ValueError(f"Expected speaker alias at {start_line + idx}: {raw.rstrip()}")
            current = line[:-1].strip()
            speakers[current] = {}
            idx += 1
            continue
        if current is None:
            raise ValueError(f"Speaker property without speaker at {start_line + idx}: {raw.rstrip()}")
        line = raw.strip()
        if ":" not in line:
            raise ValueError(f"Expected speaker key: value at {start_line + idx}: {raw.rstrip()}")
        key, value = line.split(":", 1)
        speakers[current][key.strip()] = parse_scalar(value.strip())
        idx += 1
    return speakers, idx


def parse_list(lines: list[str], idx: int) -> tuple[list[str], int]:
    items: list[str] = []
    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip():
            idx += 1
            continue
        stripped = raw.strip()
        if not raw.startswith("  ") or not stripped.startswith("- "):
            break
        items.append(str(parse_scalar(stripped[2:].strip())))
        idx += 1
    return items, idx


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.isdigit():
        return int(value)
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def extract_speaker_voice_map(meta: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, str]:
    raw_speakers = meta.get("speakers")
    if not isinstance(raw_speakers, dict) or not raw_speakers:
        errors.append(error("SPEAKER_MISSING", "Frontmatter must define speakers."))
        return {}
    speaker_voice_map: dict[str, str] = {}
    for alias, data in raw_speakers.items():
        if not isinstance(data, dict):
            errors.append(error("SPEAKER_MISSING", f"Speaker {alias} must contain properties."))
            continue
        voice = data.get("voice")
        if not voice:
            errors.append(error("SPEAKER_VOICE_MISSING", f"Speaker {alias} must define voice."))
            continue
        speaker_voice_map[str(alias)] = str(voice)
    return speaker_voice_map


def apply_speaker_voice_overrides(speaker_voice_map: dict[str, str], overrides: list[str], errors: list[dict[str, Any]]) -> None:
    for item in overrides:
        if "=" not in item:
            errors.append(error("SPEAKER_VOICE_INVALID", f"Invalid --speaker-voice value: {item}. Use Speaker1=Puck."))
            continue
        speaker, voice = item.split("=", 1)
        speaker = speaker.strip()
        voice = voice.strip()
        if not speaker or not voice:
            errors.append(error("SPEAKER_VOICE_INVALID", f"Invalid --speaker-voice value: {item}. Use Speaker1=Puck."))
            continue
        speaker_voice_map[speaker] = voice


def validate_speakers(speaker_voice_map: dict[str, str], errors: list[dict[str, Any]]) -> None:
    if len(speaker_voice_map) > 2:
        errors.append(error("TOO_MANY_SPEAKERS", "Gemini TTS supports at most two speakers.", actual=len(speaker_voice_map), limit=2))
    for speaker, voice in speaker_voice_map.items():
        if not re.match(r"^[A-Za-z0-9]+$", speaker):
            errors.append(error("INVALID_SPEAKER_ALIAS", f"Speaker alias must be alphanumeric without whitespace: {speaker}", actual=speaker))
        if voice not in GEMINI_TTS_VOICES:
            errors.append(error("INVALID_VOICE", f"Voice {voice} is not a Gemini TTS voice.", actual=voice, expected="one of GEMINI_TTS_VOICES"))


def extract_allowed_tags(meta: dict[str, Any]) -> set[str]:
    raw = meta.get("allowed_tags")
    if isinstance(raw, list):
        return {str(item).strip() for item in raw if str(item).strip()}
    return set(DEFAULT_ALLOWED_TAGS)


def extract_max_chunk_bytes(meta: dict[str, Any], warnings: list[dict[str, Any]]) -> int:
    raw = meta.get("max_chunk_bytes")
    if isinstance(raw, int) and raw > 0:
        if raw > HARD_MAX_CHUNK_BYTES:
            warnings.append(warning("MAX_CHUNK_BYTES_ABOVE_HARD_LIMIT", "max_chunk_bytes exceeds hard Gemini safety limit.", actual=raw, limit=HARD_MAX_CHUNK_BYTES))
        return raw
    return DEFAULT_MAX_CHUNK_BYTES


def split_dialogue_chunks(body: str, body_start_line: int, delimiter: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[tuple[int, str]] = []
    for offset, line in enumerate(body.splitlines(), start=body_start_line):
        if line.strip() == delimiter:
            chunks.append(make_chunk(current, len(chunks) + 1, offset))
            current = []
        else:
            current.append((offset, line))
    chunks.append(make_chunk(current, len(chunks) + 1, body_start_line + len(body.splitlines())))
    return [chunk for chunk in chunks if chunk["text"].strip() or chunk["raw_line_count"] > 0]


def make_chunk(lines: list[tuple[int, str]], number: int, delimiter_line: int) -> dict[str, Any]:
    nonempty = [(line_no, text) for line_no, text in lines if text.strip()]
    text = "\n".join(text for _, text in lines).strip()
    if nonempty:
        line_start = nonempty[0][0]
        line_end = nonempty[-1][0]
    else:
        line_start = delimiter_line
        line_end = delimiter_line
    return {
        "chunk": number,
        "line_start": line_start,
        "line_end": line_end,
        "lines": lines,
        "raw_line_count": len(lines),
        "text": text,
    }


def validate_chunk(
    chunk: dict[str, Any],
    speaker_voice_map: dict[str, str],
    allowed_tags: set[str],
    max_chunk_bytes: int,
    agent: bool,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    speakers: set[str] = set()
    tags: list[str] = []
    turn_count = 0
    text = chunk["text"]
    utf8_bytes = len(text.encode("utf-8"))
    if not text:
        errors.append(error("CHUNK_EMPTY", f"Chunk {chunk['chunk']} is empty.", chunk=chunk["chunk"], line=chunk["line_start"]))
    if utf8_bytes > max_chunk_bytes:
        errors.append(error(
            "CHUNK_TOO_LARGE",
            f"Chunk {chunk['chunk']} is {utf8_bytes} UTF-8 bytes, limit is {max_chunk_bytes}.",
            chunk=chunk["chunk"],
            line_start=chunk["line_start"],
            line_end=chunk["line_end"],
            actual=utf8_bytes,
            limit=max_chunk_bytes,
            suggested_fix=f"Split chunk {chunk['chunk']} before line {(chunk['line_start'] + chunk['line_end']) // 2}.",
        ))
    if utf8_bytes > HARD_MAX_CHUNK_BYTES:
        errors.append(error(
            "CHUNK_EXCEEDS_HARD_LIMIT",
            f"Chunk {chunk['chunk']} exceeds hard safety limit {HARD_MAX_CHUNK_BYTES} UTF-8 bytes.",
            chunk=chunk["chunk"],
            line_start=chunk["line_start"],
            line_end=chunk["line_end"],
            actual=utf8_bytes,
            limit=HARD_MAX_CHUNK_BYTES,
        ))

    for line_no, raw_line in chunk["lines"]:
        line = raw_line.strip()
        if not line:
            continue
        match = _SPEAKER_RE.match(line)
        if not match:
            if any(marker in line for marker in PROMPT_SKELETON_MARKERS):
                errors.append(error(
                    "PROMPT_SKELETON_IN_DIALOGUE_BODY",
                    "Do not paste the full Gemini prompt skeleton into the dialogue body. Put direction in frontmatter (vibe/profile) and keep body as SpeakerAlias: spoken text only.",
                    chunk=chunk["chunk"],
                    line=line_no,
                    snippet=line if agent else None,
                    suggested_fix="Move AUDIO PROFILE/SCENE/PERFORMANCE/CONTEXT into frontmatter fields, then keep only Speaker1: and Speaker2: lines in the body.",
                ))
                continue
            errors.append(error(
                "LINE_WITHOUT_SPEAKER",
                "Every non-empty dialogue line must start with SpeakerAlias: text.",
                chunk=chunk["chunk"],
                line=line_no,
                snippet=line if agent else None,
                suggested_fix="Rewrite as Speaker1: text or Speaker2: text.",
            ))
            continue
        speaker, replica = match.groups()
        turn_count += 1
        speakers.add(speaker)
        if speaker not in speaker_voice_map:
            errors.append(error("UNKNOWN_SPEAKER", f"Unknown speaker {speaker}.", chunk=chunk["chunk"], line=line_no, actual=speaker))
        if not replica.strip():
            errors.append(error("EMPTY_REPLICA", "Replica text is empty.", chunk=chunk["chunk"], line=line_no, actual=speaker))
        for tag in _TAG_RE.findall(replica):
            normalized = tag.strip()
            tags.append(normalized)
            if normalized not in allowed_tags:
                errors.append(error(
                    "INVALID_AUDIO_TAG",
                    f"Audio tag [{normalized}] is not in allowed_tags.",
                    chunk=chunk["chunk"],
                    line=line_no,
                    actual=normalized,
                    expected=sorted(allowed_tags),
                ))

    return {
        "chunk": chunk["chunk"],
        "line_start": chunk["line_start"],
        "line_end": chunk["line_end"],
        "chars": len(text),
        "utf8_bytes": utf8_bytes,
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "turn_count": turn_count,
        "audio_tags": tags,
        "text": text,
        "errors": errors,
        "warnings": warnings,
    }


def build_style_prompt(meta: dict[str, Any], speaker_voice_map: dict[str, str]) -> str:
    vibe = str(meta.get("vibe") or "Russian technical podcast. Calm, smart, warm, conversational.").strip()
    speakers = meta.get("speakers") if isinstance(meta.get("speakers"), dict) else {}
    speaker_lines = []
    for speaker, voice in speaker_voice_map.items():
        details = speakers.get(speaker, {}) if isinstance(speakers, dict) else {}
        profile = details.get("profile") if isinstance(details, dict) else None
        display_name = details.get("display_name") if isinstance(details, dict) else None
        label = f"{speaker} ({display_name})" if display_name else speaker
        speaker_lines.append(f"{label}: use voice {voice}. {profile or ''}".strip())
    return " ".join([
        "Synthesize speech for a two-speaker Russian podcast.",
        "Do not read frontmatter, instructions, delimiter lines, or speaker labels aloud.",
        "Use speaker labels only to assign voices.",
        "Respect English inline audio tags in square brackets.",
        vibe,
        " ".join(speaker_lines),
    ]).strip()


def error(code: str, message: str, **kwargs: Any) -> dict[str, Any]:
    return issue("error", code, message, **kwargs)


def warning(code: str, message: str, **kwargs: Any) -> dict[str, Any]:
    return issue("warning", code, message, **kwargs)


def issue(severity: str, code: str, message: str, **kwargs: Any) -> dict[str, Any]:
    data = {"code": code, "severity": severity, "message": message}
    for key, value in kwargs.items():
        if value is not None:
            data[key] = value
    return data
