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
| `verify-tts --audio --expected-file` | Проверить пропуски/вставки/повторы через ASR без публикации transcript | Да |
| `status --run-id` | Проверить partial/resumable run | Да |
| `concat --run-id` | Склеить существующие chunks в partial/full файл | Да |
| `history list` | Метаданные прогонов из локальной SQLite-истории (не CWD-bound) | Да |
| `history show ID` | Один прогон по внутреннему UUID или точной метке | Да |
| `history resume ID` | Возобновить нативный TTS-прогон из снимка (потенциально платно) | Да |
| `history sync ID` | Получить сохранённое состояние/результат без нового платного submit | Да |
| `history import DIR` | Безопасный offline-импорт старых каталогов (`--dry-run`) | Да |
| `history costs` | Read-only итоги расходов по валютам и операциям | Да |

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
| 60 | TTS quality failed | Существенный пропуск, посторонняя речь или повтор; human listening всё ещё обязателен |

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
| `--fallback-voice` | str | `onyx` | Принимается Polza Chat Audio только для совместимости: автоматического второго POST другим голосом нет |
| `--style-prompt` | str | — | Не поддерживается OpenRouter `/audio/speech`; явное значение отклоняется |
| `--style-prompt-file` | path | — | Не поддерживается OpenRouter `/audio/speech`; явное значение отклоняется |
| `--no-style-prompt` | flag | false | Совместимый no-op: OpenRouter всегда отправляет verbatim `input` |
| `--no-trim` | flag | false | Не обрезать финальную тишину |
| `--json` | flag | false | JSON-вывод в stdout |
| `--json-events` | flag | false | NDJSON progress events: `chunk_started`, `chunk_saved`, `chunk_failed`, `run_complete` |
| `--overwrite` | flag | false | Удалить существующую папку прогона; взаимоисключающий с `--resume` |
| `--confirm-delete-paid-audio` | flag | false | Разрешить `--overwrite` удалить подтверждённый запуск целиком, включая `chunk_*.mp3` и `raw/`; при `pending_attempt` удаление всё равно запрещено |
| `--skip-existing` | flag | false | Пропустить если прогон уже есть |
| `--resume` | flag | false | Продолжить interrupted run без повторной генерации готовых chunks; взаимоисключающий с `--overwrite` |
| `--retries` | int | `3` | Попытки на retryable provider error; для `polza-tts`/`polza-chat-audio`/`openrouter-tts` всегда 1 (локальные `qwen-local`/`omnivoice-local` сохраняют прежние значения) |
| `--retry-delay` | float | `2.0` | Начальная задержка retry в секундах |
| `--retry-max-delay` | float | `30.0` | Максимальная задержка retry |
| `--no-retry` | flag | false | Отключить retry |
| `--limit-chunks` | int | — | Сгенерировать только первые N chunks для теста |
| `--dry-run-cost` | flag | false | Посчитать chunks/chars без TTS-запросов |
| `--tts-quality-provider` | str | — | ASR для строгой сверки каждого dialogue turn до concat; обязателен для OpenRouter dialogue |
| `--tts-quality-model` | str | provider default | ASR model для quality gate |
| `--tts-quality-language` | str | — | Явный язык ASR quality gate |
| `--tts-quality-device` | str | `cpu` | ASR device |
| `--tts-quality-compute` | str | `auto` | ASR compute mode |
| `--tts-quality-runtime` | choice | `auto` | `auto`, `python`, `audio-cpp` |

### Qwen-local опции

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--mode` | choice | `preset` | `preset` (готовый голос), `auto`, `clone` (клонирование) или `design` (голос по инструкции) |
| `--qwen-instruct` | str | `QWEN_INSTRUCT` | Индивидуальная инструкция по стилю для текущего `qwen-local` прогона |
| `--sample` | str | — | Путь к референс-аудио для clone |
| `--sample-text` | str | `""` | Текст референса для clone (точнее) |

- `--mode clone` для обычного не-диалогового сценария идёт через canonical SQLite
  (нативный локальный маршрут): снимок хранит локатор референс-файла, его SHA-256
  и размер, reference text и выбранный рантайм/язык; `generate --resume` и
  `history resume ID` доказывают ту же идентичность, а референс и установленный
  локальный рантайм проверяются до первой локальной части; `history sync ID`
  модель не запускает. `preset`/`auto`/`design`, `--no-trim`, `--with-timings` и
  `--tts-quality-provider` остаются на legacy-маршруте. Подробности — в
  [Agent CLI Contract](../../../agent-cli-contract.md).

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
- `--mode design` — голос по инструкции: `--design-instruction`; для русского route accent/dialect attributes запрещены как unsupported-language conditioning. Short Russian design до 30 estimated seconds — warning + experimental; long Russian design выше threshold отклоняется до provider/GPU. Альтернативы: explicit `clone`, доступный accepted `preset`, отдельно принятые short clips или другой provider; mode/voice не подменяются.
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
| `--resume` + `--overwrite` вместе | Ошибка exit code 2 до любого удаления: флаги взаимоисключающие |
| Папка существует + `--overwrite` без chunks | Удалить папку полностью, создать заново |
| Папка существует + `--overwrite` + подтверждённые `chunk_*.mp3` | Ошибка без `--confirm-delete-paid-audio` |
| Папка существует + `--overwrite` + `pending_attempt` (неподтверждённый платный submit) | Ошибка exit code 30 даже с `--confirm-delete-paid-audio`; нужен другой `--run-id` |
| Папка существует + `--overwrite` + `run_state.json` не-объект или нечитаем | Ошибка exit code 30: состояние не доказывает отсутствие платного submit |
| Папка существует + `--skip-existing` | Вернуть `status: skipped`, не менять файлы |
| Папка существует + `--resume` | Продолжить с первого несохранённого chunk; при `pending_attempt` допускается лишь проверенный raw receipt либо известная paid media задача (см. ниже) |
| Папка существует без флагов | Ошибка exit code 30 |
| Папка — нативный прогон | Свойства: `--overwrite` отклоняется (exit 30); `--skip-existing` возвращает `skipped`; без `--resume` — ошибка; `--resume` идёт по canonical SQLite (см. ниже) |

### Нативный прогон `polza-tts` / `openrouter-tts`

Если запуск — обычный не-диалоговый `polza-tts` (модель `elevenlabs/...` через
async `/media` либо любая другая через синхронный `/audio/speech`) или
`openrouter-tts` без неподдерживаемого сочетания опций, то история становится
canonical: части, попытки, оплаченные байты и финальная
сборка хранятся в SQLite, а `run_state.json`, `chunks.json`, run/manifest JSON
пишутся как совместимый экспорт с `history_run_uuid` и `history_revision`.
Этого маршрута касаются: `--no-trim`, локальные
`--with-timings --timing-provider faster-whisper` и — отдельно — установленный
локальный `--tts-quality-provider` (`qwen-local`/`nemotron-local`); dialogue,
облачный или незарегистрированный `--tts-quality-provider`, облачные
`--timing-provider` и одновременный запрос локальных таймингов и локальной
проверки остаются на legacy JSON-writer.
Владелец каталога определяется до чтения ключа, цен и удаления: committed прогон
по `run_root` или крошечный `.voiceover-native-history.json`. Нативный след при
нечитаемой/отсутствующей БД и дескриптор без своего прогона дают fail-closed
(exit 30, `details.error_code = NATIVE_OWNERSHIP_UNVERIFIABLE` /
`NATIVE_OWNERSHIP_RUN_MISSING`) без фоллбэка на JSON. Провайдер строится
лениво: только для нового POST или GET-only добора известного Media ID;
локальная пересборка из `raw/` и починка экспортов завершённого прогона не
читают ключ. Сбой экспорта оставляет БД и аудио нетронутыми и возвращает
exit 50 с `details.error_code = NATIVE_EXPORT_FAILED`. Изменённый текст,
голос или модель блокируются до сети (`NATIVE_RESUME_IDENTITY_CHANGED`),
занятый run lock — exit 30 `NATIVE_RUN_LOCKED`. Обратный переход к legacy
JSON для такого каталога запрещён. Существующие legacy-каталоги не захватываются:
свежий native-запуск требует ещё не созданного `run_root`.

Синхронный `polza-tts` (`/audio/speech`) и `openrouter-tts` не дают
восстановимого task id. Их принятые байты связываются с БД с
`remote_task_id = null`; точная стоимость ответа `polza-tts` пишется до FFmpeg,
а `openrouter-tts` не отдаёт синхронного usage, поэтому его стоимость остаётся
неизвестной без нового сетевого GET и не выдумывается. Неопределённый
синхронный ответ оставляет маркер `submitting` и блокирует `--resume` без
повторного POST/GET; крах между записью `raw/` и его связыванием с БД
пересобирается локально по тому же receipt. В этом узком окне наблюдённая
стоимость теряется и остаётся `null`, а не ложной.

На этом маршруте `--no-trim` записывается в снимок как семантика обрезки, поэтому
`--resume` обязан повторить то же значение, иначе exit 30
`NATIVE_PROCESSING_UNSUPPORTED` до провайдера. Локальная проверка качества идёт
после готового аудио: сначала проверяется наличие установленной локальной модели
без скачивания (иначе exit 10 `NATIVE_QUALITY_MODEL_UNAVAILABLE` до платного POST),
потом существующей семантикой `verify-tts` сравнивается транскрипт с собственным
записанным текстом сценария. Приватный транскрипт хранится как
`verification_transcript` связанного прогона `verify` (`parent_uuid`), ожидаемый
текст не сохраняется, а контентно-пустой вердикт возвращается как
`quality: {complete, passed}`. Несовпадение сохраняет аудио и стоимость,
записывает наблюдённый FAIL тем же связанным прогоном `verify` (в снимке
`quality_passed: false`, без ожидаемого текста) и даёт exit 60
`NATIVE_QUALITY_FAILED`; сбой локальной транскрипции — exit 50
`NATIVE_QUALITY_ASR_FAILED`; сбой записи связанной истории — exit 50
`NATIVE_QUALITY_HISTORY_FAILED` (сохранённый FAIL не заявляется).
`history resume UUID` доигрывает только недостающую локальную проверку, а
записанный FAIL повторно сообщается тем же exit 60; `history sync UUID` её не
запускает и сообщает записанный вердикт (`quality.complete: true, passed: false`)
либо `quality.complete: false`, пока проверка не выполнена.

### Платный submit и маркер `pending_attempt`

- Платные сетевые провайдеры (`polza-tts`, `polza-chat-audio`, `openrouter-tts`)
  отправляют не более одного POST на часть от CLI; ошибка после отправки
  завершает запуск с exit code 30 без автоматического повтора.
- Перед submit CLI пишет маркер `pending_attempt` в `run_state.json` и удаляет
  его вместе с сохранением части. Маркер содержит только ограниченные `id`/
  `number`, стадию (`submitting`, `outcome_unknown`, `failed`, `raw_saved`),
  время и bounded raw receipt после сохранения ответа; на ElevenLabs `/media`
  маршруте также допускаются opaque `remote_task_id` и наблюдённая стоимость.
- Полученное платное аудио сохраняется в `raw/<chunk-id>.<mp3|wav|pcm>` до
  FFmpeg и остаётся пригодным для пересборки. Если raw receipt совпадает с
  файлом/SHA-256, identity вызова и первой незавершённой частью, `--resume`
  конвертирует его локально **без POST и Media GET**; повреждённый raw не
  разрешает повторный платный POST.
- При отсутствии пригодного raw известная принятая задача ElevenLabs `/media`
  провайдера `polza-tts` может завершиться через GET poll/download без второго
  POST, если `remote_task_id` безопасен, совпадает provider/model/voice/script,
  а все более ранние MP3 уже на диске. Даже при испорченном raw тот же валидный
  Media ID допускает только GET-only восстановление.
- Пока маркер присутствует, `--overwrite` закрыт всегда (exit code 30).
  Подтверждённый overwrite без маркера удаляет и сохранённый `raw/` только при
  явном согласии на удаление платного аудио. Если нет ни пригодного raw, ни
  допустимого Media ID, `--resume` также закрыт:
  `details.error_code = "PAID_SUBMIT_UNCONFIRMED"`. Для новой явной попытки
  используется другой `--run-id`. Чужой `id` (URL, текст запроса, секрет)
  репортится как `null`.
- Флаги `--resume` и `--overwrite` нельзя использовать вместе: CLI отклоняет их
  до удаления папки и до сети (exit code 2).
- `status --json` сообщает `can_resume: false` и `resume_block_reason:
  "paid_submit_unconfirmed"` для такого запуска, кроме проверенного raw
  receipt либо известной media-задачи с более ранними MP3 на диске (тогда
  `can_resume: true`).
  `status` не знает будущую identity вызова, поэтому `can_resume: true` не
  гарантирует, что конкретный следующий `--resume` пройдёт.
- `polza-chat-audio` не делает автоматический второй POST с `--fallback-voice`;
  выбор другого голоса требует нового явного запуска.

## Команда `history` — локальная история (S04/S05)

`voiceover history list --json`, `voiceover history show ID --json` и
`voiceover history import DIR [--dry-run] --json` работают offline без
провайдеров, ASR и платных вызовов. `--json` — в конце leaf-команды.
Импорт не меняет оригиналы.

`voiceover history resume ID --json` и `voiceover history sync ID --json`
восстанавливают один закоммиченный нативный TTS-прогон (`ID` — внутренний
UUID) из его снимка и запускают тот же executor, что и `generate --resume`;
`script.md` не перечитывается. `resume` потенциально платный: он отправляет
только действительно unattempted часть, а известный Media ID доводит GET-ами.
`sync` не делает нового платного submit: починяет JSON-экспорты завершённого
прогона, локально пересобирает raw receipt или доводит известный Media ID
GET-ами, а на unattempted части или неподтверждённом submit блокирует без
POST/GET и без чтения ключа. Оба держат тот же run lock; неизвестный/legacy
UUID и отсутствующая БД не создают ничего, `--overwrite` не принимается.
Подробности — в [Agent CLI Contract](../../../agent-cli-contract.md).

Заданный `VOICEOVER_HOME` должен быть абсолютным (иначе exit `2`); без него
используется системный data-каталог. Дом приватный (`0700`), SQLite **не зашифрована** и хранит доступный
текст. Публичный list/show выводит только метаданные без сценария, transcript,
подписанных URL и Authorization. Отсутствующая БД не создаётся; `--dry-run`
не пишет ничего и при WAL/иностранной/будущей БД сообщает неизвестный статус;
повторный импорт не дублирует расходы.

`voiceover history costs --json` — read-only глобальный отчёт: итоги по валютам
(`totals`) и разбивка по операции × валюте (`operations`), плюс `completeness` и
`local_attempts_without_api_charge`. Одна строка БД — одна уникальная
финансовая попытка, суммы накоплены `Decimal` (без float). Известная сумма не
выдаётся за exact: `exact_attempts`/`non_exact_attempts` видны, `"0"` отличимо
от `null` (unknown), отсутствующая валюта — отдельный bucket. Локальные
провайдеры из явного allowlist не считаются облачным unknown. Команда offline и
ничего не пишет: читает immutable-снимком, когда рядом нет `-wal` (и не оставляет
пустых sidecar), и по закоммиченным кадрам живого `-wal`, когда он есть. Текст,
пути, signed URL и секреты не выводятся; malformed decimal/небезопасная валюта —
fail-closed exit `30`.

Фильтры, JSON-поля, коды ошибок и денежный контракт (exact/zero/null) — в
[Agent CLI Contract](../../../agent-cli-contract.md).

## Safe defaults

| Параметр | Default | Почему |
|---|---|---|
| `--timing-device cpu` | CPU | Всегда работает |
| `--timing-compute int8` | INT8 | Минимальный RAM |
| `--timing-model small` | 486 MB | Минимальный для русского |
| Default: no overwrite | Ошибка | Защита от случайной перезаписи |
