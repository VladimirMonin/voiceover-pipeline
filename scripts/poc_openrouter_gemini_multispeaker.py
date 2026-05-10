"""POC for OpenRouter Gemini 3.1 Flash TTS prompting and multi-speaker payloads.

This script intentionally lives outside the production CLI. It sends several
small requests, saves successful generations as MP3 files, and writes a
sanitized report with request schemas and response metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from voiceover_pipeline.config import OPENROUTER_BASE_URL, get_secret
from voiceover_pipeline.media import check_media_tools, mp3_duration_ms, write_audio_as_mp3


MODEL = "google/gemini-3.1-flash-tts-preview"
OUTPUT_DIR = Path("out") / "gemini-multispeaker-poc"
TIMEOUT_SECONDS = 240


@dataclass(frozen=True)
class PocCase:
    name: str
    description: str
    body: dict[str, Any]


def main() -> int:
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not configured in the environment or .env")

    ffmpeg_path, ffprobe_path = check_media_tools()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "endpoint": f"{OPENROUTER_BASE_URL}/audio/speech",
        "output_dir": str(OUTPUT_DIR),
        "cases": [],
    }

    for case in build_cases():
        print(f"Running {case.name}: {case.description}")
        result = run_case(api_key, ffmpeg_path, ffprobe_path, case)
        report["cases"].append(result)

    report_path = OUTPUT_DIR / "poc-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")

    ok_count = sum(1 for item in report["cases"] if item["ok"])
    print(f"Successful audio generations: {ok_count}/{len(report['cases'])}")
    return 0 if ok_count else 1


def build_cases() -> list[PocCase]:
    style_prompt = (
        "Synthesize speech for a Russian technical podcast. "
        "Do not read the instructions aloud. Read only the transcript. "
        "Respect English audio tags such as [laughs], [crying], [curious], and [short pause]. "
        "Keep pronunciation clear, conversational, and natural."
    )

    dialogue_prompt = (
        "Synthesize the following Russian podcast dialogue. "
        "Speaker1 is calm, confident, warm, and slightly amused. "
        "Speaker2 is energetic, curious, and emotionally reactive. "
        "Do not read speaker labels as narration. Use them only to assign voices. "
        "Respect inline audio tags."
    )

    dialogue_text = (
        "Speaker1: Сегодня мы проверяем новый режим озвучки. [curious] Слышно ли, что это первый ведущий?\n"
        "Speaker2: [laughs] Да, и сейчас я смеюсь специально, чтобы проверить эмоциональные теги.\n"
        "Speaker1: Отлично. Теперь нужна короткая драматичная реакция.\n"
        "Speaker2: [crying] О нет, последний чанк сломался прямо перед финалом. [short pause] Но мы поймали ошибку заранее."
    )

    speaker_voice_configs = [
        {
            "speaker": "Speaker1",
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
        },
        {
            "speaker": "Speaker2",
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
        },
    ]

    camel_speaker_voice_configs = [
        {
            "speaker": "Speaker1",
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}},
        },
        {
            "speaker": "Speaker2",
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}},
        },
    ]

    return [
        PocCase(
            name="single_laugh_native_prompt",
            description="Known OpenRouter schema: one voice plus prompt and [laughs] tag.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": style_prompt,
                "input": (
                    "[curious] Это короткая проверка русского голоса Gemini. "
                    "[laughs] Сейчас должен прозвучать лёгкий смех, без лишнего текста."
                ),
            },
        ),
        PocCase(
            name="single_cry_native_prompt",
            description="Known OpenRouter schema: one voice plus prompt and [crying] tag.",
            body={
                "model": MODEL,
                "voice": "Kore",
                "response_format": "pcm",
                "prompt": style_prompt,
                "input": (
                    "[sadly] Это проверка эмоционального окраса. "
                    "[crying] Фраза должна прозвучать так, будто диктор почти плачет, "
                    "но всё ещё говорит разборчиво."
                ),
            },
        ),
        PocCase(
            name="dialogue_prompt_only_single_voice",
            description="Current schema fallback: dialogue in input, one top-level voice, roles only in prompt.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
            },
        ),
        PocCase(
            name="dialogue_top_level_multi_speaker_snake",
            description="Hypothesis: OpenRouter forwards top-level snake_case multi_speaker_voice_config.",
            body={
                "model": MODEL,
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "multi_speaker_voice_config": {"speaker_voice_configs": speaker_voice_configs},
            },
        ),
        PocCase(
            name="dialogue_top_level_multi_speaker_snake_with_voice",
            description="Hypothesis: same as snake_case, but with OpenRouter-required top-level voice string.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "multi_speaker_voice_config": {"speaker_voice_configs": speaker_voice_configs},
            },
        ),
        PocCase(
            name="dialogue_speech_config_snake",
            description="Hypothesis: OpenRouter forwards speech_config.multi_speaker_voice_config.",
            body={
                "model": MODEL,
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "speech_config": {
                    "multi_speaker_voice_config": {"speaker_voice_configs": speaker_voice_configs}
                },
            },
        ),
        PocCase(
            name="dialogue_speech_config_snake_with_voice",
            description="Hypothesis: speech_config snake_case plus OpenRouter-required top-level voice string.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "speech_config": {
                    "multi_speaker_voice_config": {"speaker_voice_configs": speaker_voice_configs}
                },
            },
        ),
        PocCase(
            name="dialogue_speech_config_camel",
            description="Hypothesis: OpenRouter forwards Google JS-style speechConfig.multiSpeakerVoiceConfig.",
            body={
                "model": MODEL,
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "speechConfig": {
                    "multiSpeakerVoiceConfig": {"speakerVoiceConfigs": camel_speaker_voice_configs}
                },
            },
        ),
        PocCase(
            name="dialogue_speech_config_camel_with_voice",
            description="Hypothesis: Google JS-style speechConfig plus OpenRouter-required top-level voice string.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "speechConfig": {
                    "multiSpeakerVoiceConfig": {"speakerVoiceConfigs": camel_speaker_voice_configs}
                },
            },
        ),
        PocCase(
            name="dialogue_voice_array",
            description="Hypothesis: OpenRouter accepts a voice array with speaker and voice pairs.",
            body={
                "model": MODEL,
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
                "voice": [
                    {"speaker": "Speaker1", "voice": "Puck"},
                    {"speaker": "Speaker2", "voice": "Kore"},
                ],
            },
        ),
        PocCase(
            name="dialogue_voice_mapping_string",
            description="Hypothesis: OpenRouter accepts voice as a compact speaker-to-voice mapping string.",
            body={
                "model": MODEL,
                "voice": "Speaker1=Puck,Speaker2=Kore",
                "response_format": "pcm",
                "prompt": dialogue_prompt,
                "input": dialogue_text,
            },
        ),
        PocCase(
            name="dialogue_casting_in_prompt_only",
            description="Hypothesis: Gemini honors per-speaker voice names from prompt while OpenRouter receives one legal voice.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": (
                    "TTS the following Russian dialogue with two clearly different voices. "
                    "Cast Speaker1 with the prebuilt Gemini voice Puck: calm, warm, thoughtful. "
                    "Cast Speaker2 with the prebuilt Gemini voice Kore: firm, bright, emotionally expressive. "
                    "Do not read speaker labels aloud. Respect [laughs], [crying], [curious], and [short pause]."
                ),
                "input": dialogue_text,
            },
        ),
        PocCase(
            name="dialogue_casting_in_input",
            description="Hypothesis: Gemini honors per-speaker voice names from a clear preamble inside input.",
            body={
                "model": MODEL,
                "voice": "Puck",
                "response_format": "pcm",
                "prompt": "Synthesize speech only. Do not read setup instructions aloud.",
                "input": (
                    "TTS the following conversation. Speaker1 uses voice Puck, calm and warm. "
                    "Speaker2 uses voice Kore, firm and expressive. Do not read speaker labels.\n\n"
                    f"{dialogue_text}"
                ),
            },
        ),
    ]


def run_case(api_key: str, ffmpeg_path: str, ffprobe_path: str, case: PocCase) -> dict[str, Any]:
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/audio/speech",
        headers={
            "Authorization": f"Bearer {api_key.removeprefix('Bearer ').strip()}",
            "Content-Type": "application/json",
        },
        json=case.body,
        timeout=TIMEOUT_SECONDS,
    )

    result: dict[str, Any] = {
        "name": case.name,
        "description": case.description,
        "ok": False,
        "status_code": response.status_code,
        "generation_id": response.headers.get("X-Generation-Id"),
        "content_type": response.headers.get("Content-Type"),
        "request_body_sanitized": case.body,
    }

    if response.status_code >= 400:
        result["error"] = response.text[:2000]
        print(f"  failed HTTP {response.status_code}")
        return result

    raw_path = OUTPUT_DIR / f"{case.name}.pcm"
    mp3_path = OUTPUT_DIR / f"{case.name}.mp3"
    raw_path.write_bytes(response.content)

    try:
        write_audio_as_mp3(ffmpeg_path, response.content, "pcm16", mp3_path)
        result["ok"] = True
        result["raw_pcm"] = str(raw_path)
        result["mp3"] = str(mp3_path)
        result["duration_ms"] = mp3_duration_ms(ffprobe_path, mp3_path)
        print(f"  ok -> {mp3_path} ({result['duration_ms']} ms)")
    except Exception as exc:  # noqa: BLE001 - POC report should capture conversion failures.
        result["error"] = f"Audio conversion failed: {exc}"
        result["raw_pcm"] = str(raw_path)
        print(f"  failed conversion: {exc}")

    return result


if __name__ == "__main__":
    raise SystemExit(main())
