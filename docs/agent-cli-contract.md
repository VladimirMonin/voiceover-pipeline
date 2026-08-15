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
| `list asr-providers` | Зарегистрированные локальные ASR-провайдеры и capabilities | да |
| `split --script` | Чанки сценария | да |
| `generate` | Полная генерация + тайминги | да |
| `timings --audio` | Тайминги из готового MP3 | да |
| `transcribe --audio` | Распознать конечный локальный аудиофайл через ASR registry | да |

Все команды можно вызвать с `--json` для машинно-читаемого вывода.

## Exit Codes

| Код | Значение | Когда |
|---|---|---|
| `0` | success | Всё ок |
| `2` | invalid args | Неверные аргументы, файл не найден, 0 чанков, неизвестный provider или неподдерживаемая capability |
| `10` | missing dependency | Не установлен выбранный локальный ASR runtime или faster-whisper |
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

Без флагов проверяет общее окружение (Polza cloud TTS baseline; faster-whisper и CUDA становятся required только с `--with-timings` или `--provider qwen-local`). Локальный ASR runtime проверяется только при явных `--with-asr --asr-provider <id>`; CUDA сама по себе не является ASR healthcheck.

С флагами проверяет конкретный workflow:

```powershell
voiceover doctor --provider qwen-local --json           # нужен CUDA
voiceover doctor --with-timings --timing-device cpu --json  # нужен faster-whisper
voiceover doctor --with-asr --asr-provider qwen-local --asr-device cpu --asr-compute auto --json
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

Для `qwen-local` используется отдельный `--qwen-instruct`. Он передаётся в
`generate_custom_voice(..., instruct=...)` только для текущего прогона. Если
флаг не указан, сохраняется прежний дефолт `QWEN_INSTRUCT` из `config.py`.

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

## `transcribe --audio --json`

`transcribe` — отдельная finite-audio ASR-команда. Она не вызывает `timings`,
не создаёт SRT и не синтезирует сегменты по всей длине аудио. Первый core slice
не реализует streaming, VAD/session lifecycle, cloud fallback или загрузку
моделей.

```powershell
voiceover list asr-providers --json
voiceover transcribe `
  --audio "recording.wav" `
  --provider local-id `
  --model "model-id" `
  --language ru `
  --device cpu `
  --compute auto `
  --json
```

- `--provider` обязателен и всегда разрешается через ASR registry. Неизвестный
  ID возвращает machine JSON error с exit code `2`; fallback в faster-whisper
  или облако запрещён.
- `--model`, `--language`, `--device` и `--compute` валидируются по capability
  выбранного provider до его factory/runtime. `qwen-local` и `nemotron-local`
  — text-only ASR IDs; неизвестный ID по-прежнему fail-closed.
- Публичных флагов `--prompt`, `--context`, `--glossary` или числового phrase
  boost нет. В API typed `ASRContextHints` различает `context_text`, glossary
  profile/digest/selected terms, `ASRPhraseHint` с силой `mild|normal|strong` и
  adapter-specific `initial_prompt`; они не записываются в стандартный receipt.
- `timestamp_mode: "none"` означает text-only ASR. `segments` или `words`
  появляются только при заявленных provider capabilities; timestamps требуют
  `native` или `forced` origin. Text-only ответ не допускается к SRT.

```json
{
  "status": "success",
  "provider": "local-id",
  "model": "model-id",
  "transcript": "...",
  "language": "ru",
  "duration_s": null,
  "source_audio": "...",
  "timestamp_mode": "none",
  "segments": [],
  "words": [],
  "execution": {
    "runtime": "...",
    "runtime_version": "...",
    "model_revision": null,
    "device": "cpu",
    "compute": "auto",
    "measurements": {}
  }
}
```

### Qwen3-ASR text-only optional runtime

`qwen-local` в ASR registry — отдельное пространство имён от одноимённого TTS
provider. Его default model — `Qwen/Qwen3-ASR-0.6B`; runtime устанавливается
явно, без автоматической загрузки модели:

```powershell
voiceover list asr-providers --json
voiceover doctor --with-asr --asr-provider qwen-local --asr-device cpu --asr-compute auto --json
```

- Runtime boundary намеренно deferred-import. Approved compatibility resolution
  declares `qwen-asr` in the `asr-qwen` optional extra; install it explicitly
  with `uv sync --extra asr-qwen`. That extra is mutually exclusive with
  `voiceover-qwen`, because their pinned Transformers runtimes are incompatible.
  CLI itself never downloads the runtime or a model.
- Adapter поддерживает finite batch audio, explicit language и typed
  `ASRContextHints.context_text`: канонические Qwen language names передаются
  как есть, а ISO-коды `de|en|es|ru` преобразуются в `German|English|Spanish|Russian`.
  Context остаётся soft contextual bias, передаваемым в
  `qwen_asr.Qwen3ASRModel.transcribe(..., context=..., language=...)`. Raw
  `--context`/`--prompt` flags и cloud fallback отсутствуют.
- Capability допускает request `--device cpu|cuda` и `--compute
  auto|bfloat16|float32`; `auto` выбирает `float32` для CPU и `bfloat16` для
  opt-in CUDA. `doctor` проверяет только импорт runtime, не наличие модели и не
  пригодность GPU.
- Adapter выдаёт transcript, effective language и execution receipt. Он не
  объявляет segment/word timestamps, forced alignment, confidence или streaming;
  результат всегда text-only (`timestamp_mode: "none"`) и не создаёт SRT.
- Отсутствующий selected runtime возвращает exit code `10` и одну remediation:
  `qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying.`

Установленный пакет, модельные веса, конкретный response schema и CPU/GPU
совместимость не доказаны этим offline slice. Они требуют отдельного
owner-approved local runtime experiment; CLI не загружает модель автоматически.

### Nemotron ASR text-only optional runtime

`nemotron-local` — отдельный local ASR ID к NVIDIA Nemotron 3.5 через Hugging
Face Transformers: `AutoProcessor` готовит audio и locale, а
`AutoModelForRNNT.generate(...)` возвращает text-only transcript. Default
identifier — `nvidia/nemotron-3.5-asr-streaming-0.6b`; он не утверждает
доступность артефакта, совместимость версии или пригодность оборудования.

```powershell
voiceover list asr-providers --json
voiceover doctor --with-asr --asr-provider nemotron-local --asr-device cpu --asr-compute auto --json
```

- Runtime boundary deferred-import: registry/listing и factory не импортируют
  `transformers`; optional extra `asr-nemotron` фиксирует совместимый runtime,
  но CLI ничего не скачивает до explicit transcription request.
- Adapter принимает finite batch audio, запрашивает `--device cpu|cuda` и только
  `--compute auto`. Для известных ISO language codes он передаёт Nemotron locale;
  `context_text`, glossary, `phrase_hints` и `initial_prompt` не передаются.
  Official model API документирует language conditioning, но не contextual bias
  или phrase boosting, поэтому обе capability не заявлены.
- Streaming, segment/word timestamps, forced alignment и confidence также не
  заявлены. Результат всегда text-only (`timestamp_mode: "none"`), не создаёт
  SRT и не является timing bridge.
- Missing selected runtime возвращает exit code `10` и одну remediation:
  `Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying.`

Этот contract покрыт mocked fixtures. Установленный Transformers runtime,
модельные веса/revision, точный response schema, phrase boosting, timestamps,
streaming, CPU/GPU compatibility и качество не проверялись; для каждого нужен
отдельный owner-approved local experiment.

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
