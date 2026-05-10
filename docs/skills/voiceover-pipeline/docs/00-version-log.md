# Версионный лог навыка

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: совместимость с приложением, история изменений, актуальность данных.

## Совместимость

| Поле | Значение |
|---|---|
| **Compatible app** | voiceover-pipeline 0.4.5 |
| **Skill revision** | 2026-05-10 |
| **Минимальная версия CLI** | 0.4.0 |
| **Максимальная проверенная** | 0.4.5 |

## Что актуально в этой версии навыка

- 4 провайдера: `polza-chat-audio`, `polza-tts`, `openrouter-tts`, `qwen-local`
- 7 протестированных моделей с реальными ценами из smoke-прогонов
- Полные списки голосов: OpenAI TTS (11), ElevenLabs через Polza (21), Gemini TTS через OpenRouter (30), Qwen preset (9)
- `list voices --json` контракт: `voices` как плоский массив + `voice_categories` объект
- ElevenLabs через Polza: async `/api/v1/media` (submit → poll → download)
- Polza TTS OpenAI: JSON base64 через `/api/v1/audio/speech`
- OpenRouter OpenAI TTS: `/audio/speech` без style_prompt
- Voiceover metadata: `format: voiceover` frontmatter для provider/model/voice/fallback/style_prompt, auto-detect in validate/generate
- OpenRouter Gemini dialogue: `--format gemini-dialogue`, frontmatter speaker map, inline audio tags, strict UTF-8 byte validation
- Gemini prompting guide: voice direction skeleton, safe audio tags, emotion recipes, voice selection, chunking limits
- Stability layer: `run_state.json`, `generation.log`, universal retries, `--resume`, paid-audio overwrite guard, `status`, `concat`, `--limit-chunks`, `--dry-run-cost`, `--json-events`
- Endpoint dispatch: `openai/*` → `/audio/speech`, `elevenlabs/*` → `/media`

## Цены (smoke 2026-04-29, не гарантия провайдера)

| Модель | Цена/мин | Валюта |
|---|---|---|
| `openai/gpt-audio-mini` | ~0.004 | RUB (anomalous) |
| `openai/gpt-audio` | ~7.00 | RUB |
| `openai/gpt-4o-mini-tts` | ~1.07 | RUB |
| `elevenlabs/text-to-speech-turbo-2-5` | ~3.51 | RUB |
| `elevenlabs/text-to-speech-multilingual-v2` | ~7.57 | RUB |
| `google/gemini-3.1-flash-tts-preview` | ~$0.030 | USD |
| `openai/gpt-4o-mini-tts-2025-12-15` | ~$0.00041 | USD |
| Qwen-local | Бесплатно | — |

## История изменений

| Дата | Изменение |
|---|---|
| 2026-05-10 | Добавлен Gemini prompting guide: `AUDIO PROFILE`/`SCENE`/`PERFORMANCE`/`CONTEXT`/`TRANSCRIPT`, safe audio tags, emotion recipes, voice selection, chunking guidance. |
| 2026-05-10 | Добавлен generic `format: voiceover` для single-speaker режимов: provider/model/voice в frontmatter, CLI overrides, full-error validator, backward compatibility с plain Markdown. |
| 2026-05-10 | Добавлен Gemini dialogue workflow: two speakers через OpenRouter `multi_speaker_voice_config` + обязательный top-level `voice`, full-error validator, `--speaker-voice`, `--agent`, chunk byte safety gates. |
| 2026-05-10 | Добавлен стабильный generation supervisor: state/log after each chunk, universal retry для всех провайдеров, безопасный `--resume`, защита paid chunks от overwrite, `status`, `concat`, `--limit-chunks`, `--dry-run-cost`. |
| 2026-05-09 | Gemini native prompt: отдельное поле `prompt` в request body вместо конкатенации в `input`. Флаги `--style-prompt-file`, `--no-style-prompt`. `prompt_mode` в manifest. Расширяемость под будущие Google/Polza модели. |
| 2026-05-01 | UV-first Python: `uv python install 3.12` вместо winget. Агент сам создаёт `.env` из `.env.example`. Remotion scene grouping: Whisper-сегменты группируются по смысловым сценам, не по чанкам. Torch CPU-only диагностика. Qwen голоса обновлены до 9 актуальных. |
| 2026-04-29 | Добавлены `polza-tts`, ElevenLabs, OpenRouter OpenAI TTS. Обновлены все цены, голоса, workflows, evaluation. |
| 2026-03 | Исходная версия навыка под voiceover-pipeline 0.3.x (Polza GPT Audio, OpenRouter Gemini, Qwen) |
