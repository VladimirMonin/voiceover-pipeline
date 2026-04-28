# Выходные артефакты: что на выходе и как читать

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: дерево out/, manifest.json, timings.json, SRT, chunks.json, связи.

## Дерево вывода

```
out/<run-id>/
├── manifest.json                            ← entry-point для агентов
├── <run-id>-voiceover-<model>.mp3           ← склеенный полный MP3
├── <run-id>-voiceover-<model>.json          ← полный run-манифест
├── <run-id>.timings.json                    ← Whisper-сегменты (ms)
├── <run-id>.srt                             ← SRT-субтитры
└── chunks/
    ├── chunk_01.mp3 … chunk_NN.mp3          ← MP3 по сценам
    └── chunks.json                          ← манифест чанков (с ценами)
```

Все JSON: UTF-8, `ensure_ascii=False`, indent=2. Null-поля не пишутся.
Цены в `chunks.json`: RUB для polza-провайдеров, USD для openrouter-tts, `null` для qwen-local.

## manifest.json — entry-point

Главный файл, который должен читать агент после генерации. Содержит пути
ко всем остальным артефактам:

```json
{
  "run_id": "prod",
  "full_mp3": "out/prod/prod-voiceover-openai-gpt-audio-mini.mp3",
  "run_json": "out/prod/prod-voiceover-openai-gpt-audio-mini.json",
  "chunks_json": "out/prod/chunks/chunks.json",
  "timings_json": "out/prod/prod.timings.json",
  "srt": "out/prod/prod.srt",
  "duration_ms": 25520
}
```

Поля `timings_json` и `srt` присутствуют только если был `--with-timings`.

**Правило:** агент читает только `manifest.json` и получает все остальные пути из него.
Не нужно угадывать имена файлов.

## .timings.json — Whisper-сегменты

Источник истины для длительностей сцен:

```json
{
  "artifact_type": "voiceover-timings",
  "model": "small",
  "backend": "faster-whisper",
  "device": "cpu",
  "compute_type": "int8",
  "language": "ru",
  "total_duration_ms": 25520,
  "segment_count": 8,
  "segments": [
    {
      "id": 0,
      "start_ms": 0,
      "end_ms": 4200,
      "duration_ms": 4200,
      "text": "Максимальное качество видео..."
    }
  ]
}
```

Поля сегмента:

| Поле | Тип | Назначение |
|---|---|---|
| `start_ms` | int | Начало в миллисекундах |
| `end_ms` | int | Конец в миллисекундах |
| `duration_ms` | int | Длительность |
| `text` | str | Whisper-транскрипция (НЕ использовать для captions!) |
| `words` | array | Word-level тайминги (только с `--word-timestamps`) |

**Для Remotion:**
- `segments[].duration_ms` → длительность каждой сцены
- `segments[].start_ms / end_ms` → синхронизация анимаций
- `words[].start_ms / end_ms` → покадровая подсветка слов

### Word-level timestamps

С `--word-timestamps` каждый сегмент получает `words[]`:

```json
{
  "words": [
    {"word": "Максимальное", "start_ms": 0, "end_ms": 860},
    {"word": "качество", "start_ms": 860, "end_ms": 1480},
    {"word": "видео", "start_ms": 1480, "end_ms": 1860}
  ]
}
```

## .srt — субтитры

Стандартный SubRip, UTF-8. Импортируется в Remotion и видеоредакторы:

```srt
1
00:00:00,000 --> 00:00:04,200
Текст первого сегмента

2
00:00:04,200 --> 00:00:08,300
Текст второго сегмента
```

## chunks.json — манифест чанков

Лежит в `chunks/`. Содержит позиции, длительности и стоимость каждого чанка:

```json
{
  "model": "openai/gpt-audio-mini",
  "voice": "ash",
  "chunk_count": 4,
  "total_duration_ms": 25520,
  "cost_total": 0.0146,
  "cost_currency": "RUB",
  "chunks": [
    {
      "number": 1,
      "id": "chunk_01",
      "file": "chunk_01.mp3",
      "duration_ms": 6400,
      "start_ms": 0,
      "end_ms": 6400,
      "text_characters": 245,
      "transcript": "...",
      "cost": 0.0032,
      "cost_currency": "RUB"
    }
  ]
}
```

Ключевые поля чанка:

| Поле | Назначение |
|---|---|
| `start_ms` / `end_ms` | Позиция в склеенном MP3 (от `ffprobe`) |
| `duration_ms` | Длительность чанка |
| `transcript` | Что модель реально произнесла |
| `cost` | Стоимость этого чанка |

## run-манифест

То же что `chunks.json` плюс:
- `main_file` — имя склеенного MP3
- `main_duration_ms` — длина склеенного файла
- `concat_method` — как склеивали
- `cost_per_minute` — цена минуты по склеенному файлу

## Связи артефактов

```
manifest.json
├── full_mp3       → <run-id>-voiceover-<model>.mp3
├── run_json       → <run-id>-voiceover-<model>.json
│   └── (содержит все поля chunks.json + main info)
├── chunks_json    → chunks/chunks.json
│   └── chunks[]   → {id, start_ms, end_ms, duration_ms, transcript, cost}
├── timings_json   → <run-id>.timings.json
│   └── segments[] → {id, start_ms, end_ms, duration_ms, text, words?}
└── srt            → <run-id>.srt
```

## Приоритет чтения для Remotion

1. `manifest.json` → все пути
2. `.timings.json` → `segments[].duration_ms` для длительности сцен
3. `.srt` → captions/subtitles
4. `chunks.json` → per-chunk alignment (`start_ms`/`end_ms`)
5. `run-*.json` → полный отчёт по стоимости (если нужен)

**Правило:** если есть `.timings.json` — используй его `duration_ms`.
НИКОГДА не оценивай длительность сцен по количеству слов (words-per-second),
если есть Whisper-тайминги.
