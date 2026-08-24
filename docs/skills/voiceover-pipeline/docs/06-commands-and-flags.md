# Команды и флаги: полный CLI-справочник

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: команды, флаги, exit codes, JSON-контракт, правила безопасности.

## Команды

| Команда | Назначение | JSON |
|---|---|---|
| `doctor` | Проверить окружение (Python, FFmpeg, ключи, Whisper, CUDA) | Да |
| `validate --script` | Проверить Markdown-сценарий | Да |
| `list providers` | Показать доступных TTS-провайдеров | Да |
| `list voices --provider` | Показать голоса провайдера | Да |
| `list timing-models` | Показать локальные Whisper-модели | Да |
| `list timing-providers` | Показать всех провайдеров распознавания (local + cloud) | Да |
| `split --script` | Разбить сценарий на чанки (без генерации) | Да |
| `generate` | Полная генерация: TTS + MP3 + опционально тайминги | Да |
| `timings --audio` | Извлечь Whisper-тайминги из готового MP3 | Да |
| `status --run-id` | Проверить partial/resumable run | Да |
| `concat --run-id` | Склеить существующие chunks в partial/full файл | Да |

Все команды поддерживают `--json` для машинно-читаемого вывода.

## Exit codes

| Код | Значение | Когда |
|---:|---|---|
| 0 | success | Всё ок |
| 2 | invalid args | Файл не найден, неверный run-id, 0 чанков |
| 10 | missing dependency | faster-whisper не установлен |
| 11 | no ffmpeg/ffprobe | FFmpeg не найден в PATH |
| 20 | no key | Нет POLZA_API_KEY или OPENROUTER_API_KEY в .env |
| 30 | provider/run error | API error, папка существует без --overwrite |
| 40 | timing blocked / whisper error | openrouter-whisper отклонён (нет таймкодов) или whisper timing упал (но MP3 сохранён!) |
| 50 | output error | Ошибка записи/удаления файлов |

## Stdout/Stderr контракт

**С `--json`:**
- `stdout`: ровно один JSON object (success или error)
- `stderr`: progress-логи и предупреждения
- exit code: семантический

**Без `--json`:**
- `stdout`/`stderr`: человекочитаемый вывод
- `stderr`: ошибки и предупреждения

## JSON output contract

### Success

```json
{"status": "success", "..."}
```

### Error

```json
{"status": "error", "error": "описание", "code": 30}
```

## Команда `generate` — все флаги

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--provider` | choice | `polza-chat-audio` | `polza-chat-audio`, `polza-tts`, `openrouter-tts`, `qwen-local`, `omnivoice-local` |
| `--model` | str | `openai/gpt-audio-mini` | ID модели |
| `--script` | path | `in/script.md` | Путь к Markdown-сценарию |
| `--delimiter` | str | `******` | Разделитель чанков |
| `--output-dir` | path | `out` | Корень выходной директории |
| `--run-id` | str | авто | Имя прогона (только `[a-zA-Z0-9._-]`) |
| `--voice` | str | зависит от провайдера | Голос |
| `--format` | choice | `markdown` | `markdown`, `voiceover`, `dialogue` или compatibility alias `gemini-dialogue` |
| `--max-chunk-chars` | int | `2000` | Validation limit для `voiceover` metadata scripts |
| `--speaker-voice` | repeat | — | Override для Gemini dialogue: `Speaker1=Puck` (можно повторять, по одному на спикера) |
| `--fallback-voice` | str | `onyx` | Запасной голос для Polza Chat Audio |
| `--style-prompt` | str | дефолтный | Стиль подачи для TTS (OpenRouter Gemini) |
| `--style-prompt-file` | path | — | Читать prompt из файла |
| `--no-style-prompt` | flag | false | Отключить prompt полностью |
| `--no-trim` | flag | false | Не обрезать финальную тишину |
| `--json` | flag | false | JSON-вывод в stdout |
| `--json-events` | flag | false | NDJSON progress events: `chunk_started`, `chunk_saved`, `chunk_failed`, `run_complete` |
| `--overwrite` | flag | false | Удалить существующую папку прогона |
| `--confirm-delete-paid-audio` | flag | false | Разрешить `--overwrite` удалить существующие `chunk_*.mp3` |
| `--skip-existing` | flag | false | Пропустить если прогон уже есть |
| `--resume` | flag | false | Продолжить interrupted run без повторной генерации готовых chunks |
| `--retries` | int | `3` | Количество попыток на retryable provider error |
| `--retry-delay` | float | `2.0` | Начальная задержка retry в секундах |
| `--retry-max-delay` | float | `30.0` | Максимальная задержка retry |
| `--no-retry` | flag | false | Отключить retry |
| `--limit-chunks` | int | — | Сгенерировать только первые N chunks для теста |
| `--dry-run-cost` | flag | false | Посчитать chunks/chars без TTS-запросов |

### Qwen-local опции

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--mode` | choice | `preset` | `preset` (готовый голос), `auto`, `clone` (клонирование) или `design` (голос по инструкции) |
| `--qwen-instruct` | str | `QWEN_INSTRUCT` | Индивидуальная инструкция по стилю для текущего `qwen-local` прогона |
| `--sample` | str | — | Путь к референс-аудио для clone |
| `--sample-text` | str | `""` | Текст референса для clone (точнее) |

### OmniVoice-local опции (локальный)

`omnivoice-local` — явный offline-only провайдер, модель
`audio-cpp/omnivoice-q8_0`. Обычная одноголосая озвучка остаётся одним native
session на прогон. `format: dialogue` создаёт один bound bank profile на turn,
при этом admission/runtime остаётся общим; два profile ID с одинаковым
`reference_sha256` отклоняются. Режимы:

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--mode` | choice | `preset` | `auto`, `preset` (bank), `clone`, `design` |
| `--voice-bank` | path | — | Путь к `catalog.json` voice bank для `--mode preset` (обязателен в preset) |
| `--reference-audio` | path | — | Референс-аудио для `--mode clone` |
| `--reference-text` | str | — | Текст референса для `--mode clone` |
| `--design-instruction` | str | — | Инструкция по голосу для `--mode design` |

- `--mode auto` — модель без voice guidance; `--voice` и reference/design флаги запрещены.
- `--mode preset` — голос из voice bank: `--voice-bank <catalog.json>` +
  опционально `--voice <profile-id>` (без `--voice` берётся `default_voice` каталога).
- `--mode clone` — ad-hoc клонирование: `--reference-audio` + `--reference-text`.
- `--mode design` — голос по инструкции: `--design-instruction`.
- `--voice` вне preset+bank, Qwen-опции и style-флаги для этого провайдера fail closed.
- `list voices --provider omnivoice-local --voice-bank <catalog.json>` показывает профили банка.

### Whisper timing опции (generate + timings)

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--with-timings` | flag | false | Запустить Whisper после TTS; dependency preflight выполняется до TTS |
| `--timing-provider` | choice | `faster-whisper` | `faster-whisper` (локально), `openrouter-whisper` (облачно, без таймкодов), `groq-whisper` (облачно, сегменты+слова), `xai-stt` (облачно, слова+confidence) |
| `--timing-model` | str | `small` / `openai/whisper-large-v3-turbo` | ID модели (зависит от провайдера) |
| `--timing-device` | choice | `cpu` | `auto`, `cpu`, `cuda` (только для faster-whisper) |
| `--timing-compute` | choice | `int8` | `auto`, `int8`, `int8_float16`, `float16`, `float32` (только faster-whisper) |
| `--timing-language` | str | `ru` | Код языка |
| `--word-timestamps` | flag | false | Word-level тайминги (только faster-whisper) |

### `validate` Gemini dialogue options

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--format` | choice | `markdown` | Включить `voiceover`, `dialogue` или alias `gemini-dialogue` валидатор |
| `--provider` | choice | frontmatter | Override provider для `voiceover` metadata |
| `--model` | str | frontmatter | Override/check model for metadata format |
| `--voice` | str | frontmatter | Override voice для `voiceover` metadata |
| `--speaker-voice` | repeat | — | Override voice map: `Speaker2=Kore` |
| `--agent` | flag | false | Добавить snippets и suggested fixes в JSON |

`voiceover` и `dialogue` (включая alias `gemini-dialogue`) валидаторы возвращают все ошибки за один прогон.
Генерация с metadata-форматом блокируется, если `valid: false`.

### Dialogue: локальные и платные флаги

- Платный путь: `--provider openrouter-tts` + `--format dialogue`
  (модель `google/gemini-3.1-flash-tts-preview`). Top-level `--voice` для
  dialogue — производная совместимость от первого спикера; явный
  конфликтующий `--voice` отклоняется до создания провайдера.
- Локальный путь: `--provider omnivoice-local --mode preset --voice-bank
  <catalog.json>`; two-speaker dialogue требует два bank profile с разными
  fingerprints. `qwen-local` не является dialogue provider.
- `--speaker-voice` (repeat) переопределяет голос спикера: `Host=Kore`,
  `Guest=Puck`. Оба спикера должны остаться с различными голосами.

## Команда `timings` — флаги

Рекомендуемый production flow: сначала `generate` для платного аудио, затем
отдельно `timings`. Это отделяет TTS от распознавания и упрощает recovery.

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--audio` | str | **обязательный** | Путь к аудио-файлу (MP3, Opus, WAV, FLAC, ...) |
| `--output-dir` | path | `out` | Корень выходной директории |
| `--run-id` | str | stem аудиофайла | Имя прогона |
| `--timing-provider` | choice | `faster-whisper` | `faster-whisper` (локально), `openrouter-whisper` (облачно, без таймкодов), `groq-whisper` (облачно, сегменты+слова), `xai-stt` (облачно, слова+confidence) |
| `--model` | str | зависит от провайдера | ID модели (см. `list timing-providers`) |
| `--device` | choice | `cpu` | `auto`, `cpu`, `cuda` (только faster-whisper) |
| `--compute` | choice | `int8` | Тип вычислений (только faster-whisper) |
| `--language` | str | `ru` | Код языка |
| `--json` | flag | false | JSON-вывод |
| `--word-timestamps` | flag | false | Word-level тайминги (только faster-whisper) |
| `--overwrite` | flag | false | Перезаписать |
| `--skip-existing` | flag | false | Пропустить |

## `list voices` — JSON контракт

```powershell
voiceover list voices --provider polza-tts --json
```

Ответ:

```json
{
  "status": "success",
  "provider": "polza-tts",
  "voices": ["alloy", "ash", "ballad", "coral", ...],
  "voice_categories": {
    "openai": ["alloy", "ash", "ballad", ...],
    "elevenlabs": ["Rachel", "Aria", "Roger", ...]
  }
}
```

- `voices` — **всегда** плоский массив (backward-compatible)
- `voice_categories` — объект с разбивкой по семействам (опционально, есть у `polza-tts` и `openrouter-tts`)

Для `openrouter-tts` категории: `"gemini"` и `"openai"`.

## `list timing-providers` — JSON контракт

```bash
voiceover list timing-providers --json
```

Ответ:

```json
{
  "status": "success",
  "timing_providers": [
    {
      "id": "faster-whisper",
      "type": "local",
      "models": [{"id": "small", "parameters_m": 244, ...}]
    },
    {
      "id": "openrouter-whisper",
      "type": "cloud",
      "currency": "USD",
      "models": [{...}]
    },
    {
      "id": "groq-whisper",
      "type": "cloud",
      "currency": "USD",
      "timestamps": ["segment", "word"],
      "models": [{...}]
    },
    {
      "id": "xai-stt",
      "type": "cloud",
      "currency": "USD",
      "timestamps": ["word"],
      "models": [{...}]
    }
  ]
}
```

- `type`: `"local"` (faster-whisper) или `"cloud"` (openrouter-whisper, groq-whisper, xai-stt)
- `timestamps`: какие таймкоды поддерживает провайдер — `segment`, `word` или поле отсутствует (openrouter-whisper — только текст)
- `models`: у local — `parameters_m`/`disk_mb`/`speed`, у cloud — `id`/`description`

## `--run-id` правила

Разрешено: `[a-zA-Z0-9._-]`, например `prod`, `prod-01`, `prod_01`, `prod.v1`.

Запрещено:
- `.`, `..`, путь с `/` или `\`
- Leading/trailing whitespace
- Trailing dot или space
- Абсолютные пути
- Windows reserved names: `CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`
- Illegal chars: `<>:"|?*` и control chars

## `--output-dir` правила

Запрещено:
- Drive root (`C:\`)
- Home directory
- CWD (current working directory)

Разрешено: относительные (`out`, `out/project`) и абсолютные пути вне CWD/home/root.

## Existing output policy

| Ситуация | Поведение |
|---|---|
| Папка не существует | Создать |
| Папка существует + `--overwrite` без chunks | Удалить папку полностью, создать заново |
| Папка существует + `--overwrite` + chunks | Ошибка без `--confirm-delete-paid-audio` |
| Папка существует + `--skip-existing` | Вернуть `status: skipped`, не менять файлы |
| Папка существует + `--resume` | Продолжить с первого несохранённого chunk |
| Папка существует без флагов | Ошибка exit code 30 |

## Safe defaults

| Параметр | Default | Почему |
|---|---|---|
| `--timing-device cpu` | CPU | Всегда работает |
| `--timing-compute int8` | INT8 | Минимальный RAM |
| `--timing-model small` | 486 MB | Минимальный для русского |
| Default: no overwrite | Ошибка | Защита от случайной перезаписи |
