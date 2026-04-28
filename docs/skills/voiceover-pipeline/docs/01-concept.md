# Концепция: Что такое voiceover-pipeline и зачем он нужен

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Это концептуальный документ: WHO, WHY, архитектура, место в media-пайплайне.

## Что это

`voiceover-pipeline` — standalone CLI, который превращает Markdown-сценарий
в пакет готовых media-артефактов: MP3-озвучку, Whisper-тайминги в миллисекундах,
SRT-субтитры и `manifest.json` как единую точку входа для агентов.

Это не просто обёртка над TTS API. Это agent-grade инструмент с:

- жёстким JSON-контрактом (`--json` → stdout содержит ровно один JSON object)
- семантическими exit codes (0/2/10/11/20/30/40/50)
- safe output policy (overwrite/skip/error)
- `manifest.json` как единый entry-point для последующей автоматизации

## Зачем он нужен агентам

Обычный TTS API даёт аудиофайл, но агенту для production-видео нужно больше:

1. **Точные тайминги.** Remotion-сцены должны начинаться и заканчиваться
   синхронно с речью. Whisper даёт реальные `start_ms/end_ms` каждого сегмента.
2. **Субтитры.** SRT-формат для импорта в видеоредакторы и Remotion.
3. **Чанки по сценам.** Markdown-сценарий разбивается на чанки через `******`,
   каждый чанк озвучивается отдельно, позиции сохраняются в `chunks.json`.
4. **Склеенный MP3.** Все чанки склеиваются в один файл для финального монтажа.
5. **Машинный контракт.** JSON-вывод, exit codes, manifest — агент не парсит
   человекочитаемый текст, а опирается на структуру.

## Архитектура

```
Markdown-сценарий (script.md)
       │
       ▼
voiceover validate --script script.md    ← валидация
       │
       ▼
voiceover generate --with-timings        ← TTS + Whisper
       │
       ├──► chunks/*.mp3                 ← MP3 по сценам
       ├──► full.mp3                     ← склеенный файл
       ├──► timings.json                 ← Whisper-сегменты (ms)
       ├──► captions.srt                 ← субтитры
       ├──► chunks.json                  ← манифест чанков
       └──► manifest.json                ← entry-point
                │
                ▼
         Remotion / монтаж / подкаст
```

## Четыре TTS-провайдера

| Провайдер | Тип | API | Валюта | Ключ |
|---|---|---|---|---|
| Polza Chat Audio | Cloud, chat-based | `/chat/completions` | RUB | `POLZA_API_KEY` |
| Polza TTS | Cloud, классический TTS + ElevenLabs | `/audio/speech` или `/media` | RUB | `POLZA_API_KEY` |
| OpenRouter TTS | Cloud, агрегатор TTS | `/audio/speech` | USD | `OPENROUTER_API_KEY` |
| Qwen-local | Local GPU | Внутрипроцессный | Бесплатно | Не нужен |

**Polza TTS** — model-aware dispatch:
- `openai/*` → `POST /api/v1/audio/speech` (JSON с base64 audio, `contentType: audio/mpeg`)
- `elevenlabs/*` → `POST /api/v1/media` (async task → poll `GET /media/{id}` → download URL)

**OpenRouter TTS** — агрегатор, поддерживает Gemini и OpenAI TTS:
- Gemini: style_prompt работает, голоса Google (30 имён)
- OpenAI TTS: style_prompt НЕ используется, голоса OpenAI (11 имён)

## Семь протестированных моделей

| # | Модель | Провайдер | Цена/мин | Валюта |
|---|---:|---|---|---:|
| 1 | `openai/gpt-audio-mini` | polza-chat-audio | ~0.004 | RUB |
| 2 | `openai/gpt-audio` | polza-chat-audio | ~7.00 | RUB |
| 3 | `openai/gpt-4o-mini-tts` | polza-tts | ~1.07 | RUB |
| 4 | `elevenlabs/text-to-speech-turbo-2-5` | polza-tts | ~3.51 | RUB |
| 5 | `elevenlabs/text-to-speech-multilingual-v2` | polza-tts | ~7.57 | RUB |
| 6 | `google/gemini-3.1-flash-tts-preview` | openrouter-tts | ~$0.030 | USD |
| 7 | `openai/gpt-4o-mini-tts-2025-12-15` | openrouter-tts | ~$0.00041 | USD |

Цены — реальные smoke-прогоны 2026-04-29, не гарантия провайдера.
Модель Qwen-local не показана — бесплатно на локальном GPU.

## Whisper Timing

Для точных таймингов используется `faster-whisper` — CPU-оптимизированная
реализация OpenAI Whisper. Модель `small` (244M параметров, ~486 MB) —
минимальная рабочая для русского языка, ~2× realtime на CPU с int8.

Тайминги первичны, точность текста вторична — для captions используется
утверждённый сценарий, а не Whisper-транскрипция.

## Аудиообработка

- **Polza Chat Audio:** Stream SSE → PCM base64 чанки → сборка → MP3 через FFmpeg (24 kHz, mono, 128 kbps)
- **Polza TTS OpenAI:** JSON base64 (`contentType: audio/mpeg`) → декодирование в MP3
- **Polza TTS ElevenLabs:** `/media` async → poll → download MP3 с URL
- **OpenRouter:** PCM 24 kHz → MP3 через FFmpeg
- **Qwen-local:** WAV → MP3 через FFmpeg
- **Обрезка тишины:** автоматическое удаление финальной тишины после речи
  (можно отключить `--no-trim`)
- **Склейка:** все чанки → один MP3 через `ffmpeg concat`

## Место в production-пайплайне

`voiceover-pipeline` находится между «есть сценарий» и «есть usable media assets».
Он НЕ рендерит видео и НЕ пишет сценарий. Его зона:

> Сценарий (Markdown) → CLI → Артефакты (MP3 + timings + SRT) → Remotion / монтаж
