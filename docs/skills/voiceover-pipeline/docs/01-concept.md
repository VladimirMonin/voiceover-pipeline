# Концепция: Что такое voiceover-pipeline и зачем он нужен

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Это концептуальный документ: WHO, WHY, архитектура, место в media-пайплайне.

## Что это

`voiceover-pipeline` — standalone CLI, который превращает Markdown-сценарий
в пакет готовых media-артефактов: MP3-озвучку, Whisper-тайминги в миллисекундах,
SRT-субтитры и `manifest.json` как единую точку входа для агентов.

Это не просто обёртка над TTS API. Это agent-grade инструмент с:

- жёстким JSON-контрактом (`--json` → stdout содержит ровно один JSON object)
- семантическими exit codes (0/2/10/11/20/30/40/50/60)
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
voiceover generate --with-timings        ← TTS + выбранный timing route (после разрешения на paid/сеть)
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

## Зарегистрированные TTS-маршруты

Облачные: `polza-chat-audio`, `polza-tts`, `openrouter-tts`; локальные без
платы провайдеру: `qwen-local`, `omnivoice-local` (требуются собственные ресурсы
и подготовленные веса). `voiceover list providers --json` показывает реестр,
а не доступность, тариф или слуховое качество внешнего API. Следуйте
`voiceover help providers.polza` и [обзору маршрутов](05-providers-and-models.md).

**Polza TTS** — model-aware dispatch:
- `openai/*` → `POST /api/v1/audio/speech` (JSON с base64 audio, `contentType: audio/mpeg`)
- `elevenlabs/*` → `POST /api/v1/media` (async task → poll `GET /media/{id}` → download URL)

**OpenRouter TTS** — текущий speech-каталог поддерживает Gemini:
- Gemini: `input` строго verbatim, голос из зарегистрированного списка
  `voiceover list voices --provider openrouter-tts --json`; эффект голоса не
  доказан без отдельной live/listening приёмки.
- исторический OpenAI Mini TTS ID больше не допускается до запроса

## История и поиск

Canonical SQLite сохраняет прогоны, попытки, наблюдённые точные стоимости и
приватные текстовые источники; JSON — совместимый экспорт. Лексический FTS5
ищет только сохранённые тексты (`voiceover help search.lexical`).
`semantic`/`hybrid` честно возвращают `SEARCH_MODE_DEFERRED` и отнесены к S09
следующего релиза, а не к текущей возможности. Исторические smoke-цены
не подтверждают будущий тариф и не дают потолок платного вызова.

## Whisper Timing

Для локальных таймингов доступен `faster-whisper`; cloud timing-маршруты —
отдельные платные вызовы (см. [ASR-справочник](13-speech-recognition-providers.md)). Интегрированный
`generate --with-timings` требует заранее закешированных весов до платного
TTS; отдельный `timings` может загрузить модель и требует разрешения на сеть.
Тайминги являются наблюдёнными границами; ASR-текст не подменяет утверждённый
сценарий автоматически.

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
