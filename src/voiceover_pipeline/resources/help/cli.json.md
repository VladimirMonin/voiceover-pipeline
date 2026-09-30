---
topic: cli.json
title: Машинный контракт: stdout, JSON и exit codes
summary: Один JSON-объект на stdout при --json, NDJSON-события отдельно, стабильные семантические коды выхода.
related: start.quick, runs.resume, providers.polza
---

## Правило вывода

С `--json` CLI печатает в stdout **ровно один** JSON-объект: либо результат, либо ошибку. Диагностика и предупреждения идут в stderr. Без `--json` stdout остаётся человекочитаемым, ошибки — в stderr.

```json
{"status": "success", "...": "..."}
```

```json
{"status": "error", "error": "описание ошибки", "code": 30,
 "details": {"error_code": "МАШИННЫЙ_КОД"}}
```

`details.error_code` — дополнительный стабильный код; `code` — числовой семантический код выхода.

## События прогресса

- `generate --json-events` печатает поток событий по одному JSON-объекту в строке (NDJSON).
- `--json` и `--json-events` взаимоисключающие: смешивать два машинных режима нельзя.

## Exit codes

| Код | Значение |
|---|---|
| `0` | success |
| `2` | invalid args: неверные аргументы, отсутствующий файл, 0 чанков, неизвестный провайдер, неподдерживаемая возможность |
| `10` | missing dependency: нет выбранного локального runtime или `faster-whisper` |
| `11` | нет `ffmpeg`/`ffprobe` в PATH |
| `20` | no key: нет `POLZA_API_KEY` или `OPENROUTER_API_KEY` |
| `30` | provider/run error: ошибка API, существующий каталог без `--overwrite` |
| `40` | whisper error: тайминги не удались |
| `50` | output error: ошибка записи/удаления или сбой сохранения уже полученного результата в историю |
| `60` | TTS quality failed: ASR-сверка нашла существенный пропуск, вставку или повтор (аудио сохранено) |

Ошибка разбора аргументов при `--json` превращается в один JSON-объект `{"status": "error", "error": "Invalid command-line arguments", "code": 2}` с exit `2`.

## Справка не зависит от окружения

`voiceover help ...` читает Markdown из установленного пакета: ключи, `.env`, FFmpeg, GPU, история и сеть не нужны, и она ничего не печатает в stdout кроме запрошенного Markdown или одного JSON-объекта. Обычный `voiceover --help` остаётся коротким списком команд.
