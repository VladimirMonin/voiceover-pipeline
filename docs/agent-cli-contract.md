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

Ошибки парсинга argparse при `--json` (например, конфликт mutually exclusive
flags) превращаются в единственный JSON error
`{"status": "error", "error": "Invalid command-line arguments", "code": 2}`
с exit code `2`.

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

Проверяет: Python, FFmpeg, FFprobe, `.env`, ключи, faster-whisper, CUDA и
явно выбранную локальную конфигурацию OmniVoice.

Без флагов проверяет общее окружение (Polza cloud TTS baseline; faster-whisper и CUDA становятся required только с `--with-timings` или `--provider qwen-local|omnivoice-local`). Локальный ASR runtime проверяется только при явных `--with-asr --asr-provider <id>`; CUDA сама по себе не является ASR healthcheck.

С флагами проверяет конкретный workflow:

```powershell
voiceover doctor --provider qwen-local --json           # нужен CUDA
voiceover doctor --provider omnivoice-local --json      # нужен CUDA и явный local model path
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

`VOICEOVER_QWEN_TTS_RUNTIME=python` — неявный и явный rollback route для
`qwen-local`. Значение `audio-cpp` явно выбирает
`AudioCppQwenTTSProvider`; эта selection не переключается автоматически.
Для Linux container route обязательна существующая локальная директория
`VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL` с `model.safetensors`, `config.json`,
`tokenizer_config.json` и пакетом `speech_tokenizer`. Опциональный
`VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON` задаёт JSON-массив argv локального
container command; default — `["docker"]`, shell-like строка недопустима.
Маршрут использует фиксированный pinned image и не принимает JSON-driver из
`VOICEOVER_AUDIO_CPP_BINARY` как production transport. Отсутствующий или
некорректный ресурс fail-closed и не возвращает Python route. Проверенный пакет
передаётся только в typed runtime JSON как `payload.model_artifact_path` вместе
с mode-specific `model_id` и `mode`; subprocess не наследует произвольное
окружение для выбора модели. Любое другое значение
`VOICEOVER_QWEN_TTS_RUNTIME` — invalid args (exit code `2`).

### `omnivoice-local`: fixed offline female style condition

`omnivoice-local` — явный offline-only provider с единственной моделью
`audio-cpp/omnivoice-q8_0`. На Linux он использует pinned audio.cpp CUDA
container, встроенное условие `female`, fixed seed и internal text chunks по 420 символов. Это не AutoVoice и не named preset/voice ID. VOP объединяет подготовленные fragments в один запрос, поэтому audio.cpp обрабатывает их в одной container/model session. Перед `generate` требуется задать
`VOICEOVER_OMNIVOICE_MODEL` на локальный Q8_0 GGUF и явно подтвердить local-only
noncommercial use: `VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE=accept-cc-by-nc-4.0-local-use`.
VOP не скачивает модель; до provider/runtime он потоково проверяет SHA-256 exact artifact.
Опциональный `VOICEOVER_OMNIVOICE_CONTAINER_COMMAND_JSON` — JSON argv для
локального Docker command (default `["docker"]`). `doctor` проверяет этот
явный file/config boundary и GPU probe, но не загружает модель и не доказывает
реальный container inference.

Распознаются глобальный `--mode preset|clone|design` (default `preset`) и
флаги `--reference-audio <path>`, `--reference-text <text>`,
`--design-instruction <text>`:

- `--mode preset` (fixed-style) — поведение не изменилось: названные
  `--voice`, `--sample`, `--sample-text`, `--qwen-instruct`, `--style-prompt`,
  `--style-prompt-file`, `--no-style-prompt`, `--fallback-voice`,
  `--speaker-voice`, а также `--reference-audio`/`--reference-text`/
  `--design-instruction` fail closed с exit code `2` (invalid args).
- `--mode clone` требует читаемый файл через `--reference-audio` и непустой
  `--reference-text`; `--design-instruction` запрещён. После валидации режим
  fail closed с exit code `2` («not implemented»): текущий fixed-style provider
  путь не реализует клонирование, и CLI отказывается использовать его как
  fallback для клона.
- `--mode design` требует непустой `--design-instruction`; `--reference-audio`/
  `--reference-text` запрещены. Аналогично fail closed с exit code `2`
  («not implemented»): голосовой дизайн не реализован текущим provider путём.
- Оба режима валидируются до построения provider; constructed provider для
  них не создаётся. Clone/design — планируемые, а не рабочие capability.

Named `--voice`, Qwen cloning/sample options и style controls для этого
provider fail closed: named preset, streaming и Qwen options не входят в
контракт. На Windows Docker/WSL не выбираются: нужен
`VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE` с рядом лежащим
`audio_cpp_dependency_closure.json`, проверяющим SHA-256 EXE/DLL closure, и
`VOICEOVER_OMNIVOICE_MODEL` и тот же noncommercial-use acknowledgment. Отсутствующий,
несовпавший SHA-256 или closure даёт unavailable route; fallback к container нет. Это
статически проверенный factory/package route, а не claim о Windows inference/readiness.
Полный pinned receipt и ограничения
лицензии: [OmniVoice Local TTS](omnivoice-local-tts.md).

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

`transcribe` — отдельная ASR-команда для конечного аудиофайла. Она не вызывает
`timings` и не создаёт SRT. Для `qwen-local` и `nemotron-local` длинная
предзаписанная запись обрабатывается автоматически: CLI измеряет источник через
`ffprobe`, извлекает последовательные ограниченные `ffmpeg`-фрагменты и
собирает их в один результат. Generic streaming/session lifecycle, cloud
fallback и загрузка моделей командой не заявлены.

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
  сохраняют свои отдельные family IDs; неизвестный ID по-прежнему fail-closed.
- `--context <text>` и `--context-file <path>` — mutually exclusive; заполняют
  typed `ASRContextHints.context_text`, передаваемый в request до provider
  probe. Blank/whitespace inline context и missing/unreadable/blank context
  file — аргументная ошибка (exit code `2`), причём до lookup provider. Значение
  не попадает в стандартный receipt и в JSON-вывод.
- `--runtime auto|python|audio-cpp` (default `auto`) — явный запрос маршрута
  ASR runtime. Explicit `--runtime audio-cpp` для всех зарегистрированных
  provider-ов сейчас fail closed с exit code `2` до dependency probe и factory:
  native audio.cpp path ещё не реализован как выбираемый CLI-маршрут, и CLI
  отказывается от fallback на другой runtime. `auto` и `python` принимаются,
  но фактический выбор runtime остаётся прежним (env-driven через
  `VOICEOVER_AUDIO_CPP_BINARY` и т.п.); `--runtime` сам по себе маршрут не
  переключает.
- Публичных флагов `--prompt`, `--glossary` или числового phrase
  boost нет. В API typed `ASRContextHints` различает `context_text`, glossary
  profile/digest/selected terms, `ASRPhraseHint` с силой `mild|normal|strong` и
  adapter-specific `initial_prompt`; они не записываются в стандартный receipt.
- `timestamp_mode: "none"` означает обычный text-only ответ adapter. `segments`
  или `words` появляются только при заявленных provider capabilities;
  timestamps имеют origin `native`, `forced` или `chunked`. `chunked` —
  консервативный span внешнего long-form фрагмента, а не claim о native/forced
  word alignment. Text-only ответ не допускается к SRT.

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

Если native ASR route запросил word timestamps, `execution` дополнительно
может содержать `raw_timestamp_entries`: неизменённые записи runtime до
нормализации в canonical VOP words. Поле отсутствует у text-only результатов.

Для `qwen-local` и `nemotron-local` источник длиннее 120 s планируется с
target 110 s в рабочем окне 90–120 s и hard maximum 120 s. Запросы с word
timestamps используют 1 s overlap и позиционную дедупликацию по абсолютным
меткам; text-only маршруты используют смежные фрагменты без overlap, чтобы не
угадывать повторяющийся текст. При доступности выбирается близкая
low-energy/silence boundary, но не раньше 90 s (кроме естественно короткого
final tail). Один provider instance вызывается последовательно с одинаковыми
typed request options на каждом фрагменте.
`duration_s` такого результата равен длительности исходника, а
`execution.long_form` добавляет проверяемые `source_duration_s`,
`covered_duration_s`, `processed_duration_s`, `coverage_verified` и manifest
каждого фрагмента (`input_*`, измеренный `output_duration_s`, его delta и
допуск, `output_status`, `coverage_*`, status и counts). Для decoded
MP3/codec seek-timebase границы допускается только bounded 0.10 s delta
от плановой длительности каждого фрагмента; большее расхождение остаётся
fail-closed. Планировщик
отклоняет gap, отсутствующий tail, выход за hard limit и известный
token/truncation signal; ошибочный `ffprobe`/`ffmpeg` boundary возвращает exit
code `11`. Последний неполный external chunk создаётся явно — VOP не использует
strict-`<` streaming loop, который мог бы молча отбросить хвост.
Text-only long-form использует смежные фрагменты без audio-overlap: повторяющиеся
слова на соседних границах сохраняются как распознанная речь, а не удаляются
эвристически. Word-timed маршруты сохраняют 1 s overlap и дедуплицируют только
по доказанным позиционным меткам.

### Qwen3-ASR local optional runtime

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
  `--prompt` flag и cloud fallback отсутствуют; публичные `--context`/
  `--context-file` заполняют только `context_text` (см. `transcribe` выше).
  Glossary и phrase hints публичными флагами не выбираются.
- Capability допускает request `--device cpu|cuda` и `--compute
  auto|bfloat16|float32`; `auto` выбирает `float32` для CPU и `bfloat16` для
  opt-in CUDA. Python route допускается только с уже размещёнными official
  model and Hugging Face cache directories under `/media/v/storage`; it passes
  these paths and `local_files_only=True` to the runtime, so it cannot download
  or use a root cache. `doctor` checks that local admission without loading a
  model or assessing GPU suitability.
- Для короткого input adapter выдаёт transcript, effective language и execution
  receipt. Для long-form public CLI выполняет описанную выше external
  orchestration, поэтому Qwen не является short-audio-only route. Text-only
  long result получает `chunked` segment spans; word request uses separately
  admitted local official ForcedAligner, merges absolute canonical words and
  сохраняет `alignment_origin="forced"`. Confidence и generic streaming не
  заявлены.
- Отсутствующий selected runtime возвращает exit code `10` и одну remediation:
  `qwen-asr runtime is unavailable. Install an approved qwen-asr runtime before retrying.`
- На Windows optional audio.cpp Qwen ASR route требует
  `VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE`, рядом лежащий checksummed
  `audio_cpp_dependency_closure.json`, `VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL`
  и `VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL`. Docker/WSL fallback не
  выбирается; отсутствие package/model closure остаётся unavailable. Этот
  static contract не доказывает Windows inference/readiness.
- При отсутствии required local model/cache directories under `/media/v/storage`
  selected Python route also returns exit code `10`, without a network attempt.

Установленный пакет, модельные веса, конкретный response schema и CPU/GPU
совместимость не доказаны этим offline slice. Они требуют отдельного
owner-approved local runtime experiment; CLI не загружает модель автоматически.

### Nemotron ASR: Python fallback и opt-in audio.cpp native timestamps

`nemotron-local` сохраняет один public family ID для NVIDIA Nemotron 3.5.
Без `VOICEOVER_AUDIO_CPP_BINARY` factory выбирает существующий deferred-import
Python/Transformers adapter; при явном binary route выбирается pinned
`audio.cpp` adapter. Default identifier —
`nvidia/nemotron-3.5-asr-streaming-0.6b`; он не утверждает доступность
артефакта, совместимость версии или пригодность оборудования.

```powershell
voiceover list asr-providers --json
voiceover doctor --with-asr --asr-provider nemotron-local --asr-device cpu --asr-compute auto --json
```

- Registry/listing и factory остаются deferred-import. Optional extra
  `asr-nemotron` нужен только Python fallback; CLI ничего не скачивает до
  explicit transcription request. При выбранном audio.cpp route dependency probe
  проверяет только configured driver boundary.
- Adapter принимает finite batch audio, `--device cpu|cuda`, `--compute auto`
  и language. В audio.cpp route он передаёт language без локального mapping в
  `--language`; pinned Nemotron session выбирает integer prompt ID из
  `prompt_dictionary` processor config модели. VOP не отправляет prompt ID,
  task или request-side prompt dictionary. Пустой language оставляет source
  default; неизвестный language отклоняется source. Это model conditioning, а
  не свободный context prompt.
- `timestamp_mode: "word"` в audio.cpp route включает `--words-out` и возвращает
  native RNN-T tokenizer entries. SentencePiece/metaspace chunks детерминированно сливаются
  в canonical words; punctuation остаётся у слова, нулевые spans допустимы,
  confidence остаётся `null`. Raw entries сохраняются в receipt до такого
  слияния. Python fallback остаётся text-only.
- `context_text`, glossary, `phrase_hints` и `initial_prompt` fail closed в
  audio.cpp route. Публичный `--context`/`--context-file` для Python fallback
  не влияет на transcribe: fallback не передаёт context в модель.
  Пиннутый offline wire contract не доказывает phrase boosting:
  hotword/term extension остаётся capability-unavailable, пока не появятся
  decoder and live term evidence. В long-form public CLI Nemotron также
  использует external bounded orchestration: Python text-only fallback получает
  `chunked` segments, а native audio.cpp words сохраняют normalised native
  timestamps с absolute offsets. Поэтому Nemotron не является short-audio-only
  route; generic streaming, forced alignment и confidence по-прежнему не
  заявлены.
- `transcribe` с missing selected Python runtime возвращает exit code `10` с
  remediation `Nemotron ASR runtime is unavailable. Install an approved Hugging Face Transformers runtime before retrying.`
  Missing selected audio.cpp route в `transcribe` также возвращает exit code `10` с remediation
  `audio.cpp Nemotron ASR runtime is unavailable. Set VOICEOVER_AUDIO_CPP_BINARY to the pinned JSON driver before retrying.`
  `doctor --with-asr` не запускает модель: он помечает ASR provider unavailable
  и добавляет ту же remediation в JSON `warnings`.

Этот contract покрыт mocked fixtures. Не выполнялись реальный audio.cpp binary,
модельные веса/revision, RNN-T decoder parity, phrase boosting, streaming,
CPU/GPU compatibility и качество таймкодов; для них нужен отдельный approved
offline/local runtime experiment.

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
- Qwen-local и omnivoice-local требуют CUDA GPU
- OmniVoice local uses the Linux container route or a statically checked native-Windows factory; native Windows inference/readiness is not claimed
- OmniVoice local `--mode clone|design` и ASR explicit `--runtime audio-cpp` проходят валидацию, но fail closed (exit code `2`) как not implemented — планируются, не являются рабочими capability
- Первый Whisper запуск скачивает модель (~486 MB) из HuggingFace
- При `--with-timings` ошибка Whisper — hard failure (code 40), но MP3 уже сохранён
