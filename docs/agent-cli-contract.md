# Agent CLI Contract

Контракт для агентов, работающих с `voiceover-pipeline`. Предсказуемый JSON-ввод/вывод, стабильные exit codes, карта артефактов.

## Команды

| Команда | Зачем | JSON |
|---|---|---|
| `doctor` | Проверить окружение | да |
| `validate --script` | Проверить сценарий | да |
| `list providers` | Доступные TTS-провайдеры | да |
| `list voices --provider X` | Голоса провайдера | да |
| `list timing-models` | Whisper-модели | да |
| `split --script` | Чанки сценария | да |
| `generate` | Полная генерация + тайминги | да |
| `timings --audio` | Тайминги из готового MP3 | да |

Все команды можно вызвать с `--json` для машинно-читаемого вывода.

## Exit Codes

| Код | Значение | Когда |
|---|---|---|
| `0` | success | Всё ок |
| `2` | invalid args | Неверные аргументы, файл не найден, 0 чанков |
| `10` | missing dependency | faster-whisper не установлен |
| `11` | no ffmpeg/ffprobe | FFmpeg не найден в PATH |
| `20` | no key | Нет POLZA_API_KEY или OPENROUTER_API_KEY |
| `30` | provider/run error | API error, папка существует без --overwrite |
| `40` | whisper error | Whisper timing не удался |
| `50` | output error | Ошибка записи/удаления файлов |

## Stdout/Stderr Contract

**--json:**
- `stdout`: ровно один JSON object (success или error)
- `stderr`: progress-логи и предупреждения
- exit code: семантический код из таблицы

**Без --json:**
- `stdout`/`stderr`: человекочитаемый вывод
- `stderr`: ошибки и предупреждения

При `--json` в stdout никогда не должно быть не-JSON строк.

## JSON Output Contract

### Success

```json
{
  "status": "success",
  "...": "..."
}
```

### Error

```json
{
  "status": "error",
  "error": "описание",
  "code": 30
}
```

## `doctor --json`

Проверяет: Python, FFmpeg, FFprobe, `.env`, ключи, faster-whisper, CUDA.

Без флагов проверяет общее окружение (Polza cloud TTS baseline; faster-whisper и CUDA становятся required только с `--with-timings` или `--provider qwen-local`).

С флагами проверяет конкретный workflow:

```powershell
voiceover doctor --provider qwen-local --json           # нужен CUDA
voiceover doctor --with-timings --timing-device cpu --json  # нужен faster-whisper
```

```json
{
  "status": "success",
  "required_ok": true,
  "optional_ok": false,
  "workflow_ok": true,
  "checks": {
    "python": {"ok": true, "version": "3.14.2", "required": true},
    "ffmpeg": {"ok": true, "path": "...", "required": true},
    "ffprobe": {"ok": true, "path": "...", "required": true},
    "env_file": {"ok": true, "path": "...", "required": true},
    "polza_key": {"ok": true, "required": true},
    "openrouter_key": {"ok": false, "required": false},
    "faster_whisper": {"ok": true, "required": false},
    "cuda": {"ok": false, "required": false}
  },
  "warnings": [
    "CUDA is unavailable: qwen-local and cuda timings will not work, but cloud TTS and CPU timings are OK."
  ]
}
```

Агент опирается на `workflow_ok` для принятия решения. Отсутствие CUDA не блокирует cloud TTS и CPU timings.

## `validate --script --json`

```json
{
  "status": "success",
  "valid": true,
  "chunks": 2,
  "total_chars": 370,
  "issues": [],
  "warnings": []
}
```

При `issues` агент предлагает пользователю исправить сценарий.

## `generate` — Style Prompt Flags

| Флаг | Тип | Default | Поведение |
|---|---|---|---|
| `--style-prompt` | str | дефолтный | Prompt строкой из CLI |
| `--style-prompt-file` | path | — | Читать prompt из файла |
| `--no-style-prompt` | flag | false | Отключить prompt полностью |

Приоритет: `--no-style-prompt` > `--style-prompt-file` > `--style-prompt` > дефолт из config.py.

## `generate --json` (output)

```json
{
  "status": "success",
  "provider": "polza-chat-audio",
  "model": "openai/gpt-audio-mini",
  "run_id": "prod",
  "files": {
    "full_mp3": "...",
    "run_json": "...",
    "chunks_json": "...",
    "manifest_json": "...",
    "timings_json": "...",
    "srt": "..."
  },
  "duration_ms": 25520,
  "segment_count": 8,
  "cost": {
    "total": 0.0146,
    "currency": "RUB"
  }
}
```

Агент читает `files.manifest_json` как entry-point или напрямую `files.timings_json` для таймингов.

## `timings --audio --json`

```json
{
  "status": "success",
  "files": {
    "timings_json": "...",
    "srt": "..."
  },
  "segment_count": 8,
  "duration_ms": 25520
}
```

### `generate --json` (skipped)

```json
{
  "status": "skipped",
  "reason": "run folder exists",
  "run_id": "prod",
  "files": {...}
}
```

### `generate --json` (timing failure)

Timing failure при `--with-timings` — это hard error (code 40), но MP3 сохранён:

```json
{
  "status": "error",
  "error": "Voiceover generated but timing extraction failed: ...",
  "code": 40
}
```

MP3 можно восстановить отдельно: `voiceover timings --audio ...`

## Артефакты

### Карта файлов

```
out/<run-id>/
├── manifest.json                    ← entry-point
├── <run-id>-voiceover-<model>.mp3   ← полный MP3
├── <run-id>-voiceover-<model>.json  ← run-манифест
├── <run-id>.timings.json            ← Whisper тайминги
├── <run-id>.srt                     ← SRT субтитры
└── chunks/
    ├── chunk_01.mp3 … chunk_NN.mp3
    └── chunks.json                  ← манифест чанков
```

### Приоритет для Remotion

1. `.timings.json` → `segments[].start_ms, end_ms, duration_ms` → scene durations
2. `.srt` → captions
3. `chunks.json` → `chunks[].start_ms, end_ms, transcript` → per-chunk alignment

## Safe Defaults

| Флаг | Дефолт | Зачем |
|---|---|---|
| `--timing-device cpu` | CPU | Всегда работает |
| `--timing-compute int8` | INT8 | Минимальный RAM |
| `--timing-model small` | 486 MB | Минимальный для русского |
| Дефолт: no overwrite | Ошибка | Защита от случайной перезаписи |

## `--run-id` Rules

Разрешено: `[a-zA-Z0-9._-]`, например `prod`, `prod-01`, `prod_01`, `prod.v1`.

Запрещено:
- `.`, `..`, путь с `/` или `\`
- leading/trailing whitespace
- trailing dot or space
- абсолютные пути
- Windows reserved names: `CON`, `PRN`, `AUX`, `NUL`, `COM1`..`COM9`, `LPT1`..`LPT9`
- illegal chars: `<>:"|?*` и control chars

## `--output-dir` Rules

Запрещено:
- drive root (`C:\`)
- home directory
- current working directory

Разрешено: относительные пути (`out`, `out/project`) и абсолютные пути внутри файловой системы вне CWD/home/root.

## Existing Output Policy

| Ситуация | Поведение |
|---|---|
| Папка не существует | Создать |
| Папка существует + `--overwrite` | Удалить папку полностью, создать заново |
| Папка существует + `--skip-existing` | Вернуть `status: skipped`, файлы не менять |
| Папка существует без флагов | Ошибка exit code 30 |

## Agent Workflow (Golden Path)

```powershell
# 1. Проверить окружение
voiceover doctor --provider polza-chat-audio --with-timings --json

# 2. Проверить сценарий
voiceover validate --script "script.md" --json

# 3. Сгенерировать озвучку + тайминги
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "script.md" `
  --run-id "prod" `
  --with-timings `
  --word-timestamps `
  --json `
  --overwrite

# 4. Прочитать артефакты
#    manifest.json    → все пути
#    .timings.json    → scene durations (ms)
#    .srt             → captions
#    chunks.json      → per-chunk alignment
```

## Known Limitations

- Whisper text может содержать ошибки — используй утверждённый сценарий для captions, Whisper только для timing
- `--word-timestamps` подходит для visual highlights, но не гарантирует семантически точных границ слов
- Cloud prices are snapshots из API на момент прогона, не гарантия
- Qwen-local требует CUDA GPU
- Первый Whisper запуск скачивает модель (~486 MB) из HuggingFace
- При `--with-timings` ошибка Whisper — hard failure (code 40), но MP3 уже сохранён
