# Артефакты и анализ

Что получается после каждого прогона: файлы, JSON, цены. И сравнение семи моделей.

> **Образцы аудио:** первый чанк от каждой модели (OGG Vorbis 24 kHz mono):
> - [polza-gpt-audio-mini-chunk-01.ogg](polza-gpt-audio-mini-chunk-01.ogg) — GPT Audio Mini, ~0.004 ₽/мин (anomalous)
> - [polza-gpt-audio-chunk-01.ogg](polza-gpt-audio-chunk-01.ogg) — GPT Audio, ~7.00 ₽/мин
> - [polza-openai-gpt-4o-mini-tts-chunk-01.ogg](polza-openai-gpt-4o-mini-tts-chunk-01.ogg) — Polza GPT-4o Mini TTS, ~1.07 ₽/мин
> - [polza-elevenlabs-turbo-2-5-chunk-01.ogg](polza-elevenlabs-turbo-2-5-chunk-01.ogg) — Polza ElevenLabs Turbo 2.5, ~3.51 ₽/мин
> - [polza-elevenlabs-multilingual-v2-chunk-01.ogg](polza-elevenlabs-multilingual-v2-chunk-01.ogg) — Polza ElevenLabs Multilingual v2, ~7.57 ₽/мин
> - [openrouter-gemini-tts-chunk-01.ogg](openrouter-gemini-tts-chunk-01.ogg) — OR Gemini TTS, ~$0.030/мин
> - [openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg](openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg) — OR GPT-4o Mini TTS, ~$0.00041/мин

## Что на выходе

```text
out\<run-id>\
├── chunks\
│   ├── chunk_01.mp3
│   ├── ...
│   ├── chunk_12.mp3
│   └── chunks.json               ← манифест чанков
├── <run-id>-voiceover-<модель>.mp3   ← склеенный файл
├── <run-id>-voiceover-<модель>.json  ← полный манифест прогона
├── <run-id>.timings.json             ← Whisper-сегменты (ms)
├── <run-id>.srt                      ← субтитры SRT
└── manifest.json                     ← entry-point для агентов
```

Все JSON — UTF-8, кириллица как есть, отступ 2 пробела. Поля со значением `null` не пишутся.

## `chunks.json`

Лежит в `chunks/`. Содержит всё про чанки **без** информации о склеенном файле.

| Поле | Что это |
|---|---|
| `model` | ID модели |
| `voice` | Голос |
| `style_prompt` | Стилевой промпт (только Gemini) |
| `chunk_count` | Сколько чанков |
| `total_duration_ms` | Суммарная длина |
| `cost_total` | Общая стоимость |
| `cost_currency` | Валюта (`RUB` или `USD`) |
| `cost_per_minute` | Цена минуты |
| `pricing_snapshot` | Цены модели на момент прогона |
| `chunks` | Массив чанков (см. ниже) |

### Поля чанка

| Поле | Что это |
|---|---|
| `number` | Номер по порядку |
| `id` | `chunk_01`, `chunk_02`, ... |
| `file` | Имя MP3-файла |
| `duration_ms` | Длина в миллисекундах |
| `start_ms`, `end_ms` | Позиция в склеенном файле |
| `text_characters` | Длина текста в символах |
| `transcript` | Что модель реально произнесла |
| `generation_id` | ID у провайдера |
| `cost` | Стоимость (float) |
| `cost_rub` | Стоимость в рублях (только Polza) |
| `cost_exact` | Точная стоимость строкой от провайдера |
| `cost_currency` | Валюта (`RUB` или `USD`) |
| `usage` | Токены/usage от провайдера |

## `<run-id>-voiceover-*.json`

То же что `chunks.json` плюс:

| Поле | Что это |
|---|---|
| `main_file` | Имя склеенного MP3 |
| `main_duration_ms` | Длина склеенного файла |
| `concat_method` | Как склеивали |
| `cost_per_minute` | Цена минуты (по склеенному файлу) |

## `.timings.json`

Схема Whisper-таймингов:

| Поле | Что это |
|---|---|
| `model` | Whisper модель |
| `backend` | `faster-whisper` |
| `device` | `cpu` / `cuda` |
| `compute_type` | `int8` / `int8_float16` / `float16` |
| `language` | Код языка |
| `segment_count` | Сколько сегментов |
| `total_duration_ms` | Суммарная длительность |
| `segments` | Массив сегментов (см. ниже) |

### Поля сегмента

| Поле | Что это |
|---|---|
| `id` | Номер сегмента |
| `start_ms` / `end_ms` | Тайминг в миллисекундах |
| `duration_ms` | Длительность |
| `text` | Текст сегмента |
| `words` | Word-level тайминги (если `--word-timestamps`) |

## `.srt`

Стандартный SubRip, UTF-8. 3 блока: номер, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, текст, пустая строка. Импортируется в Remotion и видеоредакторы.

## `manifest.json`

Entry-point для агентов. Содержит пути ко всем артефактам:

| Поле | Что это |
|---|---|
| `run_id` | Имя прогона |
| `full_mp3` | Путь к склеенному MP3 |
| `run_json` | Путь к run-манифесту |
| `chunks_json` | Путь к chunks-манифесту |
| `timings_json` | Путь к Whisper-таймингам (если есть) |
| `srt` | Путь к SRT (если есть) |
| `duration_ms` | Длительность аудио |

## Обработка аудио

### Конвертация в MP3

Разные провайдеры отдают аудио по-разному:

- **Polza `/audio/speech`**: JSON с base64-аудио (`contentType: audio/mpeg`). Пайплайн декодирует и пишет MP3 напрямую.
- **Polza `/media` (ElevenLabs)**: async task, результат — URL для скачивания MP3 с S3.
- **OpenRouter**: raw PCM bytes. Пайплайн конвертирует через FFmpeg:
  ```
  ffmpeg -f s16le -ar 24000 -ac 1 -i pipe:0 -codec:a libmp3lame -b:a 128k файл.mp3
  ```
- **Qwen-local**: WAV через soundfile. Пайплайн конвертирует в MP3 через FFmpeg.

Параметры MP3: 24 000 Гц, моно, 128 kbps (64 kbps для Qwen).

### Обрезка тишины

Некоторые модели оставляют длинную тишину после речи. Пайплайн:

1. Находит все silent-сегменты через `ffmpeg silencedetect`
2. Смотрит на последний сегмент
3. Если он доходит до конца файла (зазор ≤ 0.35 сек) и длиннее 1.5 сек — обрезает

Внутренние паузы не трогаются. Отключить: `--no-trim`.

### Склейка

```
ffmpeg -f concat -safe 0 -i список.txt -codec:a libmp3lame -b:a 128k full.mp3
```

Перекодировка (а не stream copy) — чтобы файл был timestamp-safe для монтажа.

## Откуда берутся цены

**Polza:**
- Снимок цен: `GET /api/v1/models`
- Точная стоимость: `GET /api/v1/history/generations/{id}` → поле `clientCost` (RUB)

**OpenRouter:**
- Снимок цен: `GET /api/v1/models?output_modalities=speech`
- Точная стоимость: `GET /api/v1/generation?id={generation_id}` → поле `total_cost` (USD)
- Пайплайн жмёт до 4 раз с паузой 3 сек — OpenRouter не сразу отдаёт usage

## Сравнение прогонов

Один и тот же сценарий (2 чанка, 370 символов), все модели:

| Модель | Длина | Стоимость | Цена/мин | Валюта |
|---|---:|---:|---|---|
| `openai/gpt-audio-mini` | 209.3 сек | 0.015 | ~0.004 | RUB (anomalous) |
| `openai/gpt-audio` | 31.7 сек | 3.70 | ~7.00 | RUB |
| `openai/gpt-4o-mini-tts` | 33.4 сек | 0.59 | ~1.07 | RUB |
| `elevenlabs/text-to-speech-turbo-2-5` | 28.5 сек | 1.67 | ~3.51 | RUB |
| `elevenlabs/text-to-speech-multilingual-v2` | 26.4 сек | 3.33 | ~7.57 | RUB |
| `google/gemini-3.1-flash-tts-preview` | 26.7 сек | 0.014 | ~0.030 | USD |
| `openai/gpt-4o-mini-tts-2025-12-15` | 32.3 сек | 0.00022 | ~0.00041 | USD |

## Связь артефактов

```
manifest.json
├── full_mp3 → <run-id>-voiceover-<model>.mp3 (склеенный файл)
├── run_json → <run-id>-voiceover-<model>.json (полный манифест, включает chunks)
├── chunks_json → chunks/chunks.json (отдельные чанки, позиции, transcript)
├── timings_json → <run-id>.timings.json (Whisper сегменты ms)
└── srt → <run-id>.srt (субтитры SRT)
```

Агент читает только `manifest.json` и получает все остальные пути.

## Где лежат готовые артефакты

```text
C:\PY\voiceover-pipeline\out\
├── <your-run-id>\
│   ├── manifest.json
│   ├── *.mp3, *.json, *.timings.json, *.srt
│   └── chunks/
└── ...
```
