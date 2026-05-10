# Рабочие сценарии (workflows)

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: готовые end-to-end цепочки для типовых задач.
> Полный список моделей и голосов: `docs/05-providers-and-models.md`.

## Fresh Project Bootstrap (нет ничего)

Когда проект пустой — ни `.env`, ни `script.md`, ни `out/`:

1. **Установи пререквизиты.** Проверь Python/UV/FFmpeg, если нет — поставь сам.
   Если среда не позволяет — попроси пользователя (см. `docs/02-install.md`).
   Если на любом шаге команда не найдена — `docs/09-troubleshooting.md`.
2. **Выбери сборку.** Production → `voiceover-pipeline[timing-whisper]`.
   Запусти без установки: `uvx voiceover-pipeline doctor` или установи постоянно: `pipx install "voiceover-pipeline[timing-whisper]"`.
3. **Создай `.env.example`.** Из шаблона `examples/env-example.md`.
4. **Создай `.gitignore`.** Если файла нет — создай. Добавь строку `.env`.
5. **Создай `script.md`.** Если сценария нет — создай из шаблона `examples/minimal-script.md`.
6. **Создай `out/`.** Папка для артефактов, будет использована как `--output-dir`.
7. **Попроси ключи.** Создай `.env` из `.env.example` сам, попроси ОДИН раз вписать ключи в `.env`.
   Больше не спрашивать. Проверить через `voiceover doctor --provider <X> --json`.
8. **Дальше — выбери workflow ниже.**

## Golden Cloud Workflow — Polza Chat Audio (рубли, chat-based)

```powershell
voiceover doctor --provider polza-chat-audio --with-timings --json
voiceover validate --script "script.md" --json
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "script.md" `
  --run-id "prod" `
  --json `
  --resume
voiceover timings --audio "out/prod/prod-voiceover-openai-gpt-audio-mini.mp3" --run-id "prod" --json --overwrite
```

## Golden Cloud Workflow — Polza TTS (рубли, классический TTS)

OpenAI TTS через Polza — JSON base64, быстро:

```powershell
voiceover doctor --provider polza-tts --with-timings --json
voiceover validate --script "script.md" --json
voiceover generate `
  --provider polza-tts `
  --model "openai/gpt-4o-mini-tts" `
  --voice "ash" `
  --script "script.md" `
  --run-id "prod" `
  --json `
  --resume
```

ElevenLabs через Polza — async `/media`, чистое качество:

```powershell
voiceover generate `
  --provider polza-tts `
  --model "elevenlabs/text-to-speech-turbo-2-5" `
  --voice "Rachel" `
  --script "script.md" `
  --run-id "elevenlabs" `
  --json `
  --resume
```

## Golden Cloud Workflow — OpenRouter (доллары)

Gemini TTS — западные голоса, style prompt:

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Puck" `
  --script "script.md" `
  --run-id "gemini-prod" `
  --json `
  --resume
```

OpenAI TTS через OpenRouter — самый дешёвый TTS в долларах:

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "openai/gpt-4o-mini-tts-2025-12-15" `
  --voice "alloy" `
  --script "script.md" `
  --run-id "openai-or" `
  --json `
  --resume
```

С style prompt (Gemini):

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Kore" `
  --style-prompt "Энергичный голос ведущего: громкий, быстрый, уверенный." `
  --script "script.md" `
  --run-id "podcast-ep1" `
  --json `
  --resume
```

С длинным prompt из файла (WVM-ассеты, expressive, stutter):

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Zephyr" `
  --style-prompt-file "prompts/expressive-narrator.txt" `
  --script "script.md" `
  --run-id "wvm-expressive" `
  --json `
  --resume
```

Без prompt (чистый Gemini TTS):

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Puck" `
  --no-style-prompt `
  --script "script.md" `
  --run-id "gemini-clean" `
  --json `
  --resume
```

## Qwen Local Workflow (бесплатно, GPU)

Требуется NVIDIA GPU + CUDA + extras `voiceover-qwen`:

```powershell
voiceover doctor --provider qwen-local --json
voiceover generate `
  --provider qwen-local `
  --mode preset `
  --voice "Aiden" `
  --script "script.md" `
  --run-id "qwen-prod" `
  --json `
  --resume
```

Clone-голос (нужен референс-аудиофайл):

```powershell
voiceover generate `
  --provider qwen-local `
  --mode clone `
  --sample "my_voice_sample.mp3" `
  --sample-text "Текст референса." `
  --script "script.md" `
  --run-id "my-voice" `
  --json `
  --resume
```

## Timings из готового MP3

Когда MP3 уже есть, а нужны только тайминги:

```powershell
voiceover timings `
  --audio "path/to/audio.mp3" `
  --output-dir "out" `
  --run-id "timed-audio" `
  --model small `
  --device cpu `
  --compute int8 `
  --language ru `
  --word-timestamps `
  --json `
  --overwrite
```

## Безопасный повторный запуск

```powershell
voiceover generate ... --resume            # продолжить безопасно
voiceover generate ... --skip-existing     # пропустить если есть
voiceover generate ... --run-id "prod-02"  # новый run-id
voiceover generate ... --overwrite --confirm-delete-paid-audio  # удалить paid chunks явно
```

## Интеграция с Remotion (полный поток)

```text
1. Установка:
   pip install "voiceover-pipeline[timing-whisper]"

2. Проверка:
   voiceover doctor --provider polza-chat-audio --with-timings --json

3. Генерация аудио:
   voiceover generate --provider polza-chat-audio --model "openai/gpt-audio-mini" --script "script.md" --run-id "production" --json --resume

4. Тайминги:
   voiceover timings --audio "out/production/production-voiceover-openai-gpt-audio-mini.mp3" --run-id "production" --word-timestamps --json --overwrite

5. Чтение артефактов:
   manifest = json.load(open("out/production/manifest.json"))
   timings = json.load(open(manifest["timings_json"]))

6. Использование в Remotion:
   for seg in timings["segments"]:
       scene = {
           "start_ms": seg["start_ms"],
           "end_ms": seg["end_ms"],
           "duration_ms": seg["duration_ms"],
           "narration": seg["text"],
           "words": seg.get("words")
       }
```

## Provider selection table

| Задача | Провайдер | Модель | Цена |
|---|---|---|---|
| Самый дешёвый, рубли | Polza Chat Audio | `openai/gpt-audio-mini` | ~0.004 RUB/мин |
| Классический TTS, рубли | Polza TTS | `openai/gpt-4o-mini-tts` | ~1.07 RUB/мин |
| Чистый голос, рубли | Polza TTS | `elevenlabs/text-to-speech-turbo-2-5` | ~3.51 RUB/мин |
| Лучшее качество речи | Polza TTS | `elevenlabs/text-to-speech-multilingual-v2` | ~7.57 RUB/мин |
| Качество интонаций | Polza Chat Audio | `openai/gpt-audio` | ~7.00 RUB/мин |
| Самый дешёвый TTS, USD | OpenRouter | `openai/gpt-4o-mini-tts-2025-12-15` | ~$0.00041/мин |
| Западные голоса | OpenRouter | `google/gemini-3.1-flash-tts-preview` | ~$0.030/мин |
| Бесплатно, GPU | Qwen-local | CustomVoice | Бесплатно |
