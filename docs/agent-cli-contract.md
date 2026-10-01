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
| `verify-tts --audio --expected-file` | Fail-closed проверка пропусков, посторонней речи и повторов через локальный ASR | да |
| `history list` | Метаданные прогонов из локальной SQLite-истории | да |
| `history show ID` | Один прогон по внутреннему UUID или точной метке | да |
| `history resume ID` | Возобновить нативный TTS-прогон из сохранённого снимка (потенциально платно) | да |
| `history sync ID` | Получить сохранённое состояние/результат известного нативного прогона без нового оплаченного submit (для известного remote id возможен GET, не POST) | да |
| `history import DIR [--dry-run]` | Безопасный offline-импорт старых `out/<run-id>` каталогов | да |
| `search QUERY` | Offline FTS5-поиск по сохранённым сценариям, транскрипциям и меткам; режим — явный `--mode` или `[search] default_mode` | да |
| `index status\|build\|rebuild` | Состояние, добор и полная пересборка производного поискового индекса | да |

Все команды можно вызвать с `--json` для машинно-читаемого вывода.

## Exit Codes

| Код | Значение | Когда |
|---|---|---|
| `0` | success | Всё ок |
| `2` | invalid args | Неверные аргументы, файл не найден, 0 чанков, неизвестный provider или неподдерживаемая capability |
| `10` | missing dependency | Не установлен выбранный локальный ASR runtime или faster-whisper |
| `11` | no ffmpeg/ffprobe | FFmpeg не найден в PATH |
| `20` | no key | Нет POLZA_API_KEY/OPENROUTER_API_KEY/GROQ_API_KEY/X_AI_API_KEY, либо непригодный явный `--env-file` |
| `30` | provider/run error | API error, папка существует без --overwrite |
| `40` | whisper error | Whisper timing не удался |
| `50` | output error | Ошибка записи/удаления файлов или сбой сохранения уже полученного результата в локальную историю (`details.error_code` = `HISTORY_PERSISTENCE_FAILED`) |
| `60` | TTS quality failed | ASR-сверка нашла существенный пропуск, вставку или повтор; аудио и receipt сохранены |

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

## Глобальный `--env-file` и разрешение секретов

`--env-file PATH` — глобальная опция **перед** подкомандой:
`voiceover --env-file PATH <command> ...`. Runtime-значение секрета разрешается
в одном детерминированном порядке:

1. непустая переменная окружения текущего процесса — приоритет; содержимое
   env-файла при этом не читается;
2. явный `--env-file PATH` (проверяется только по метаданным как regular file);
3. `<call-time CWD>/.env` — для совместимости.

Поиска `.env` по родительским каталогам нет, и путь не захватывается на момент
импорта: шаг 3 разрешается в момент вызова. Явный `--env-file` **заменяет**
рабочий `.env`, а не дополняет его: если в явном файле ключа нет, фоллбэка на
`<CWD>/.env` не происходит.

При обращении к ключу отсутствующий, не-regular (каталог) или недоступный для
проверки метаданных явный путь fail-closed с exit `20` **даже если** в процессе
есть пригодный ключ. Если процессного ключа нет, дополнительно проверяется
чтение/декодирование файла; ошибка тоже даёт exit `20`. Сообщения фиксированы и
не содержат путь или значение (`Explicit --env-file is missing or not a regular
file.` / `Explicit --env-file could not be read.`). Когда процессное значение
побеждает, содержимое файла не читается, поэтому его читаемость не проверяется.
В `--json` это тот же error envelope без пути и секретов. Read-only `help` не
проверяет файл; `doctor` отдельно сообщает о проблеме в checks без exit `20`.

`help` не читает ключ и env-файл даже при заданном `--env-file`. `doctor`
проверяет наличие ключей и отдаёт `checks.env_file.path` (разрешённый явный путь
или `<CWD>/.env`) и `checks.env_file.ok` по `is_file()`, но никогда не значение;
при непригодном явном файле `doctor` завершается с exit `0` и JSON
`status: "success"`, но сообщает `checks.env_file.ok: false` при
`required: true`, поэтому `required_ok: false` и `workflow_ok: false`;
stat-ошибка не эхоится. Подробный порядок и примеры — `voiceover help
start.quick`, `voiceover help providers.polza`.

## `help` — упакованная атомарная справка (S10)

```bash
voiceover help [TOPIC] [--raw | --json]
```

Темы — обычные Markdown-файлы в `voiceover_pipeline/resources/help/*.md`
установленного пакета; они читаются через `importlib.resources`. Рабочий каталог,
репозиторий, `.env`, ключи, FFmpeg, GPU, история и сеть для `help` не нужны — то
же самое при установленном колесе вне checkout.

- Без `TOPIC` печатается тема `index`, которая перечисляет все упакованные темы;
  отдельного флага `--list` нет.
- `--raw` печатает ровно упакованный Markdown без frontmatter и без ANSI;
  `--json` — один объект `{status, topic, title, summary, related, markdown}`;
  по умолчанию тот же Markdown с заголовком темы. `--raw` и `--json`
  взаимоисключающие (exit `2`).
- Имя темы — строгий lowercase dotted identifier (`speech.parts`). `../`, путь с
  `/`, абсолютный путь, `.env`, uppercase — usage error exit `2`
  `HELP_INVALID_TOPIC`; корректное, но отсутствующее имя — exit `2`
  `HELP_UNKNOWN_TOPIC`. Непригодный или несогласованный упакованный каталог
  (пропавшая/дублирующая тема, незнакомый frontmatter-ключ, битая `related`-
  ссылка, посторонний файл, отсутствие default-темы) — exit `30`
  `HELP_RESOURCE_ERROR`. Все эти сообщения не включают путь, содержимое файла или
  секрет.
- Frontmatter ограничен полями `topic`, `title`, `summary`, `related`; тема
  `index` обязана ссылаться на каждую другую тему.
- Справка и контракт не дублируют исчерпывающие списки провайдеров, моделей и
  флагов: актуальные значения берутся из parser metadata и `voiceover list ...`.
  `voiceover --help` остаётся кратким списком команд.

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

## `generate` / `validate`: `speech-parts`, короткая реплика и `--audio-format` (S06)

`--format speech-parts` — новый строгий YAML-формат: `version: 1`, `format: speech-parts`,
опциональный общий `vibe`, непустой упорядоченный `parts` со
 ровно одной непустой строкой `voice` и непустым `text`, опциональным
 per-part `vibe`. Дубликатов ключей, неизвестных ключей, вложенных блоков и
 tab-отступов нет: loader отклоняет их до любой работы; BOM (`utf-8-sig`)
 принимается. `provider`/`model` в документе не дублируются — только CLI или
 постоянные defaults приложения. Голоса и оба vibe остаются в YAML, поэтому CLI
 `--voice`/`--vibe`/`--style-prompt` для `speech-parts` отклоняются вместо
 скрытого override.

```bash
voiceover validate --script ./podcast.yaml --format speech-parts \
  --provider polza-tts --model google/gemini-3.8-flash-tts --json
voiceover generate --text "Добрый вечер." --voice Kore --vibe "Спокойный ведущий." --json
```

- `generate --text ... --voice ... --vibe ...` создаёт ровно одну часть. Вызов без
  `--text` сохраняет прежний выбор default script; явные `--text` и `--script`
  взаимоисключающие. `--vibe` — свободная строка, не enum: с обычным
  `--script`/legacy-форматом она отклоняется до чтения ключа/POST
  (`VIBE_UNSUPPORTED_FORMAT`, exit `2`), а в `speech-parts` vibe задаётся в YAML.
- Итоговая инструкция части = общий vibe, затем пустая строка, затем vibe части;
  отсутствующие значения пропускаются. В произносимый `text` инструкция не
  попадает и в снимке хранится отдельно (`vibe_shared`/`vibe_specific`/
  `vibe_effective`). `history resume`/`sync` восстанавливают прогон из снимка, не
  перечитывая исходный файл.
- До первого POST проверяются **все** части: `request_chars = len(text) +
  len(effective_vibe) + len(required_text_wrapper) <= 5000` (приложение-side
  консервативная политика, не доказанный предел Gemini). Поздняя over-limit
  часть не отправляет ни одного запроса (JSON `SPEECH_PART_TOO_LONG`, exit `2`).
- Маршрут `speech-parts` идёт только через DB-first нативную генерацию; legacy
  fallback отсутствует. Каталог Polza и ограниченные реальные пробы подтвердили
  модель `google/gemini-3.8-flash-tts`, endpoint `/audio/speech`, отдельный
  scalar `voice` на запрос, WAV вместо запрошенного MP3 и опубликованные
  компоненты цены. Но схема Polza не описывает два голоса в одном POST;
  инструкция со вторым голосом дала на слух один голос, а поле `instructions`
  не документировано для Gemini. Общий MP3 с тремя голосами в исследовательском
  прогоне потребовал отдельных POST по частям, не подтверждая stable API или
  внешний счёт (см. `docs/reports/2026-10-01-s06-gemini38-live-probes.md`).
  Flash-Lite не проверялся. Поэтому candidate-маршрут Flash (и любая непустая
  vibe на любом другом маршруте) по умолчанию fail-closed: JSON `error_code`
  `BLOCKED_PROVIDER_CONTRACT`, exit `30`, **до** чтения ключа и любого POST.
  Наблюдённый ID каталога не является подтверждённым speech-parts-контрактом.
- `--allow-experimental-gemini-speech-parts` — **явный opt-in** к эмпирическому
  экспериментальному маршруту `polza-tts/google/gemini-3.8-flash-tts`. Флаг
  принимается только с `--text` или `--format speech-parts`, только с этой
  provider/model парой и только на нативном DB-first маршруте; иначе — usage
  error до ключа и POST. Он не подтверждает Gemini-контракт и не делает модель
  stable: `list` её не показывает, а каждый POST по-прежнему несёт ровно один
  scalar `voice`. Эффективная инструкция части уходит отдельным полем
  `instructions` (для Gemini не документировано), а произносимый `input` —
  ровно `text`, без инструкции и без смешивания. Маршрут не обещает, что
  инструкция применена или не прочитана вслух. Флаг записывается в
  `config_snapshot.output.experimental_speech_parts`; `generate --resume` без
  того же флага отказывает, а `history resume`/`sync` продолжают ровно
  записанную политику. Принятое тело ответа и receipt сохраняются до разбора,
  наблюдённый WAV конвертируется в обычный MP3-чанк, а редиректы не следуются:
  3xx — это один наблюдённый ответ, а не повторный платный POST.
- `--audio-format {mp3,wav}` задаёт контейнер итогового merged-файла (default
  `mp3`); промежуточные части в `chunks/` остаются MP3. `wav` пишет реальный
  RIFF/WAVE через ffmpeg и допускается только на нативном маршруте; смена
  формата на `--resume` отклоняется. `speech-parts` не запускает скрытую
  облачную ASR-проверку на каждую реплику.

`validate --format speech-parts --json` возвращает `parts`, `request_chars`,
`route.admitted` и `route.reason`; синтаксическая/бюджетная ошибка — exit `2`,
неподтверждённый маршрут — предупреждение `BLOCKED_PROVIDER_CONTRACT` при
валидном документе. С `--allow-experimental-gemini-speech-parts` тот же отчёт
возвращает `route.admitted: true`; `validate` никогда не отправляет запрос.

## `generate` — Style Prompt Flags

| Флаг | Тип | Default | Поведение |
|---|---|---|---|
| `--style-prompt` | str | дефолтный | Prompt строкой из CLI |
| `--style-prompt-file` | path | — | Читать prompt из файла |
| `--no-style-prompt` | flag | false | Отключить prompt полностью |

Приоритет: `--no-style-prompt` > `--style-prompt-file` > `--style-prompt` > дефолт из config.py.

Для `openrouter-tts` style prompt не является synthesis capability:
`input` всегда равен точному произносимому тексту, `prompt_mode=none`, отдельного
`prompt` нет. Явные `--style-prompt`/`--style-prompt-file` отклоняются до billing;
frontmatter `style_prompt` получает `STYLE_PROMPT_IGNORED`.

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

### `omnivoice-local`: offline OmniVoice with auto / preset / clone / design

`omnivoice-local` — явный offline-only provider с единственной моделью
`audio-cpp/omnivoice-q8_0`. На Linux он использует pinned audio.cpp CUDA
container; на Windows — native `audiocpp_cli.exe` factory. Fixed seed `1234`
и internal text chunks по 420 символов. VOP объединяет подготовленные
fragments в один запрос, поэтому audio.cpp обрабатывает их в одной
model session (голос постоянен на весь script). Перед `generate` требуется
задать `VOICEOVER_OMNIVOICE_MODEL` на локальный Q8_0 GGUF и явно подтвердить
local-only noncommercial use:
`VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE=accept-cc-by-nc-4.0-local-use`.
VOP не скачивает модель; до provider/runtime он потоково проверяет SHA-256
exact artifact. Опциональный `VOICEOVER_OMNIVOICE_CONTAINER_COMMAND_JSON` —
JSON argv для локального Docker command (default `["docker"]`). `doctor`
проверяет этот явный file/config boundary и GPU probe, но не загружает модель
и не доказывает реальный container inference.

Распознаются глобальный `--mode auto|preset|clone|design` (default `preset`)
и флаги `--reference-audio <path>`, `--reference-text <text>`,
`--design-instruction <text>`, `--voice-bank <catalog.json>`:

- `--mode auto` — модель без voice guidance; `--voice` и reference/design
  флаги запрещены (exit code `2`).
- `--mode preset` — требует `--voice <id>` и `--voice-bank <catalog.json>`.
  Резолвит profile из каталога (пути строго внутри bank root, SHA-256
  сверяется с файлом, reference — mono WAV) и выполняет нативный clone по
  reference. Неизвестный `--voice`, невалидный каталог, несовпадение
  digest дают exit code `2`; при `--resume` с изменённым fingerprint — exit
  code `30`. Public metadata: kind `bank-preset` + `voice_id` +
  `voice_fingerprint` (sha256). Reference path/transcript не публикуются.
  Одноголосый preset-bank прогон сохраняется в каноническую историю (см.
  «Нативный локальный OmniVoice monologue»).
- `--mode clone` требует читаемый файл через `--reference-audio` и непустой
  `--reference-text`; `--design-instruction` и `--voice` запрещены.
  Reference нормализуется до PCM16 mono 24 kHz при staging. Public kind:
  `reference-clone`. Resume-identity детерминирована (sha256), изменённый
  fingerprint отклоняется с exit code `30`.
- `--mode design` требует непустой `--design-instruction` из allowlist
  (gender/age/pitch/style/accent); `--reference-audio`/`--reference-text`/
  `--voice` запрещены. Неизвестный токен fail closed с exit code `2`.
  Accent attributes относятся только к English synthesis, Chinese dialect
  attributes — только к Chinese synthesis. Текущий OmniVoice route фиксирован
  на русском языке, поэтому обе категории отклоняются до запуска модели.
  Public kind: `design-instruction`. Upstream обучал Voice Design только на
  Chinese/English. Поэтому Russian/non-English/non-Chinese design до
  provider/model/GPU admission получает warning как experimental при оценке не
  более 30 секунд и отклоняется выше upstream long-form threshold. Оценка
  выполняется offline по 2.5 words/second. Clone и preset/voice-bank не
  блокируются; policy допускает English/Chinese design, хотя текущий CLI route
  остаётся фиксированным на русском. Windows не объявляется сломанным.
  Автоматической смены mode/voice нет.

Long unsupported design возвращает exit code `2`. В `--json` обычные
`status`/`error`/`code` дополняются объектом `details`:

```json
{
  "error_code": "OMNIVOICE_DESIGN_UNSUPPORTED_LONG_LANGUAGE",
  "provider": "omnivoice-local",
  "mode": "design",
  "language": "ru",
  "estimated_duration_seconds": 30.4,
  "threshold_seconds": 30.0,
  "alternatives": [
    {
      "id": "omnivoice-clone",
      "provider": "omnivoice-local",
      "mode": "clone",
      "requires": "Russian reference audio and its transcript",
      "experimental": false
    },
    {
      "id": "omnivoice-preset",
      "provider": "omnivoice-local",
      "mode": "preset",
      "requires": "an available accepted voice-bank profile",
      "experimental": false
    },
    {
      "id": "short-design-clips",
      "provider": "omnivoice-local",
      "mode": "design",
      "requires": "separate acceptance of every clip at or below thirty estimated seconds",
      "experimental": true
    },
    {
      "id": "other-tts-provider",
      "provider": null,
      "mode": null,
      "requires": "explicit selection of another TTS provider",
      "experimental": false
    }
  ],
  "automatic_fallback": false
}
```

Каждый новый `run_state.json` и итоговый run receipt содержит content-free
`execution_source`: package version, `editable-checkout`/`installed-wheel`, Git
revision и dirty flag для editable checkout, плюс SHA-256 точного package tree.
Локальные пути и исходный текст туда не входят.

## Оплаченный submit: `--retries`, `--resume`, `--overwrite` и `status`

Платный сетевой TTS (`polza-tts`, `polza-chat-audio`, `openrouter-tts`)
отправляет не более одной попытки на часть от CLI. `--retries` больше не даёт
разрешение повторить платный submit: после входа в submit любая ошибка (таймаут,
обрыв соединения, транзиентный HTTP-ответ, всё ещё pending media-задача)
завершает запуск с exit code `30` вместо повторной отправки. Ни один платный
провайдер не делает автоматический второй POST сам: `polza-chat-audio` больше
не переигрывает часть вторым голосом из `--fallback-voice` при отказе первого —
`--fallback-voice` принимается только для совместимости, а выбор другого голоса
требует нового явного запуска. Локальные `qwen-local` и `omnivoice-local`
сохраняют прежние `--retries`/`--no-retry`/`--retry-delay`.

Перед платным submit CLI атомарно записывает в `run_state.json` маркер попытки
`pending_attempt` и удаляет его той же записью, которой сохраняет завершённую
часть. Маркер содержит только `id`/`number` части, стадию (`submitting`,
`outcome_unknown`, `failed`, `raw_saved`), время, ограниченное raw-подтверждение
после сохранения ответа и — на ElevenLabs `/media` маршруте — ограниченный opaque
`remote_task_id` вместе с уже наблюдённой стоимостью; текст запроса, signed URL
и тело ошибки в него не попадают. `outcome_unknown` означает неопределённый
платный исход, `raw_saved` — полученные байты с проверяемым локальным receipt.

Маркер блокирует, пока физически присутствует в состоянии: успешно сохранённая
часть удаляет маркер той же атомарной записью, поэтому маркер рядом с
`completed`-записью — устаревшее или конфликтующее доказательство, а не
разрешение повторить submit. Любой присутствующий маркер (включая не-объект или
маркер с чужим `id`/`number`) закрывает overwrite и `--resume` fail-closed,
кроме валидного raw-подтверждения или известной paid media задачи ниже; в
диагностику попадают только ограниченные `chunk_id`, `chunk_number`,
`attempt_status`, а `attempt_status = "unknown"` помечает нераспознанный или
повреждённый маркер. `chunk_number` репортится только когда это положительное
сгенерированное число не больше `1_000_000`; boolean, ноль, отрицательное или
слишком большое значение заменяется на `null`. `chunk_id` репортится только
тогда, когда он совпадает со сгенерированной формой для этого числа —
`chunk_{n:02d}`, `turn_{n:04d}` или `chunk_01_omnivoice_session` при `number = 1`;
любой другой `id` (`chunk_sk_live_secret`, signed URL, текст запроса, секрет)
заменяется на `null`, поэтому ошибка не может утечь чужие данные.

`--resume` и `--overwrite` взаимоисключающие и проверяются до любого удаления,
создания провайдера и сети: `--resume` продолжает существующий запуск, а
`--overwrite` сначала удаляет его, поэтому их сочетание возвращает exit code `2`
(обычный error envelope `status`/`error`/`code`) и не трогает каталог, аудио и
состояние. Обычный `--resume` или намеренный `--overwrite` без маркера работают
как прежде.

`generate --resume` проверяет маркер до необязательных preflight'ов качества
TTS и таймингов, чтения ключа, создания провайдера, запроса цен и preflight'ов
идентичности диалога, поэтому недоступный ключ, нехватка зависимости или ошибка
цен не могут подменить `PAID_SUBMIT_UNCONFIRMED`. Если маркер присутствует без пригодного raw или известного Media ID, resume
fail-closed:

- exit code `30` и обычный error envelope `status`/`error`/`code`;
- `details.error_code = "PAID_SUBMIT_UNCONFIRMED"` с `chunk_id`, `chunk_number`,
  `attempt_status`;
- провайдер не вызывается, второго POST нет.

После принятого ответа любого платного TTS (`polza-tts`, `polza-chat-audio`,
`openrouter-tts`) исходное аудио атомарно записывается в каталог запуска
`raw/<chunk-id>.<mp3|wav|pcm>` **до FFmpeg** и не удаляется после успешной части.
В маркере `raw_saved` остаются только ограниченные формат/путь/SHA-256/
generation ID и наблюдённая стоимость (если есть). `--resume` проверяет файл,
хеш, identity запуска и готовность более ранних частей, затем пересобирает
первую незавершённую часть локально **без нового POST или Media GET**. Сбой
конвертации не лишает запуск оплаченных байтов; отсутствие/повреждение raw не
разрешает повторный платный POST. Сеть и локальный диск не атомарны вместе:
если ответ ещё не записан, прежний блок сохраняется.

Другой допустимый путь — известная paid media задача `polza-tts` на
ElevenLabs-маршруте `/media`. Если маркер привязан к сгенерированному
`id`/`number` части и содержит ограниченный opaque `remote_task_id`
(`[A-Za-z0-9_-]{1,128}`, записанный атомарно до первого poll), `generate
--resume` доводит эту часть только GET-запросами (`GET /media/{id}` → download
signed URL) и никогда не отправляет второй POST, но лишь когда одновременно
выполнено всё: provider/model/voice/script identity совпадают с состоянием,
marker-часть — первый незавершённый chunk сценария, и все более ранние
`chunk_*.mp3` физически лежат на диске. Точная стоимость из completed-poll
сохраняется в маркере до скачивания signed URL, поэтому GET-only recovery без
`usage` переиспользует уже наблюдённую сумму; маркер удаляется той же атомарной
записью, что сохраняет завершённую часть. Если raw-подтверждение есть, оно
имеет приоритет над Media GET. При повреждённом raw, но валидном ID той же
Media-задачи разрешены только GET
poll/download, не второй POST. Без пригодного raw **и** допустимого Media ID,
при несовпадении identity, отсутствии более раннего MP3 или нечитаемом
состоянии `--resume` возвращает `PAID_SUBMIT_UNCONFIRMED` до чтения ключа,
preflight'ов и цен. Ни в маркер, ни в публичную диагностику не попадают signed
URL, тело запроса, произвольный `usage` или текст ошибки провайдера.

`generate --overwrite` проверяет тот же маркер до удаления каталога и до
создания провайдера. Запуск с присутствующим маркером не удаляется даже вместе
с `--confirm-delete-paid-audio`: неподтверждённый оплаченный исход остаётся
доказательством. Вместо перезаписи CLI возвращает exit code `30`, тот же
`details` и предлагает явно другой `--run-id` для новой попытки, а маркер и
файлы старого запуска сохраняются. Проверка маркера идёт до проверки
`--confirm-delete-paid-audio`, поэтому запуск, где уже сохранён первый
`chunk_*.mp3` и оставлен неподтверждённый маркер, возвращает
`PAID_SUBMIT_UNCONFIRMED`, а не общую ошибку удаления платного аудио; обычный
`--confirm-delete-paid-audio` по-прежнему разрешает удалить подтверждённый
прогон без маркера: он удаляет весь каталог, включая сохранённый `raw/`, то
есть владелец явно принимает потерю оплаченного исходника. Существующий
`run_state.json`, который не
читается как JSON-объект (невалидный JSON, `null`, список, скаляр), не может
доказать отсутствие платного submit и трактуется так же, как неподтверждённый
маркер: overwrite возвращает exit code `30` и не удаляет каталог. Корректный
старый объект без `pending_attempt` перезаписи не блокирует.

`status --json` возвращает `can_resume: false` с ограниченной машиночитаемой
причиной `resume_block_reason`: `"paid_submit_unconfirmed"` при присутствующем
маркере и `"run_state_unreadable"` при существующем `run_state.json`, который
не читается как JSON-объект (невалидный JSON, `null`, список, скаляр). Обе
причины имеют приоритет над обычным расчётом resume и выводятся одним
parseable-object без эха сырого JSON или секретов. При отсутствующем состоянии
поведение не меняется, а без блокировки `resume_block_reason` равно `null`.

Исключение: `status --json` возвращает `can_resume: true` без
`resume_block_reason` для пригодного raw-подтверждения (существующий файл с
совпадающим SHA-256) любого платного TTS либо для валидной известной paid media
задачи (`provider: polza-tts`, ElevenLabs `/media`, bounded `remote_task_id`).
Маркер не должен быть рядом с `completed` той же части; все более ранние MP3
должны лежать на диске. `status` не знает будущую identity вызова: он не проверяет
provider/model/voice/script, с которыми позднее будет запущен `--resume`, поэтому
`can_resume: true` не гарантирует, что конкретный следующий `--resume` пройдёт.

Маркер существует только в состоянии своего запуска: новый запуск с другим
`--run-id` начинает новое состояние. Восстановление по известному remote ID
входит в контракт только для описанного выше случая `polza-tts`/`/media` и не
проверено на live-провайдере: форма id и `usage` из реального ответа пока не
подтверждены. Пригодный raw receipt восстанавливает конвертацию для всех
платных TTS без provider-запросов; отдельный флаг разрешённого повторного
платного POST в контракт не входит.

## Нативная история для обычного Polza / OpenRouter TTS

Слой native-истории переводит на canonical SQLite обычные
не-диалоговые запуски `polza-tts` — как async `--model elevenlabs/...`
(`/media`), так и синхронный `/audio/speech` (любая другая модель) — обычный
не-диалоговый `polza-chat-audio` и
`openrouter-tts`. Такой запуск допускается и для plain Markdown, и для
`format: voiceover`: voiceover-валидатор уже разрешил один provider/model/voice
на весь сценарий, а снимок хеширует подготовленный текст частей, поэтому на
часть не выдумывается ни провайдер, ни голос. Этого же маршрута касаются:
записанная семантика обрезки
(`--no-trim` либо обрезка по умолчанию), интегрированные
`--with-timings` для локального `faster-whisper` **или** платных облачных
`groq-whisper`/`xai-stt` (последние идут через тот же платный boundary, что и
standalone `timings`), и установленный локальный
`--tts-quality-provider` (`qwen-local`, `nemotron-local`), дающий аудио и
локальную проверку одной командой; тайминги и локальная проверка
могут быть запрошены вместе в одном прогоне. Облачный
`--tts-quality-provider` (`xai-stt`) допускается **только** на двух диалоговых
маршрутах (см. ниже) и на не-диалоговом маршруте по-прежнему игнорируется legacy
executor'ом; `openrouter-whisper` (не даёт реальных таймстемпов) и
незарегистрированный quality-провайдер остаются вне среза. Все прочие
маршруты, все существующие legacy-каталоги и уже начатые JSON-прогоны
продолжают использовать прежний executor и прежний JSON-writer без изменений.

- **Раннее владение.** `generate` разрешает владельца каталога до legacy JSON
  recovery, до чтения ключа, до построения провайдера, до цен и до удаления.
  Владельцем считается committed история-прогон по canonical `run_root` **или**
  маленький run-local дескриптор `.voiceover-native-history.json`. Наличие
  любого нативного следа (дескриптор, `native_history` в старом
  `run_state.json`, `raw/*.receipt.json`) при недоступной/повреждённой или
  отсутствующей БД, а также дескриптор без своего прогона, дают fail-closed
  (exit `30`, `details.error_code` = `NATIVE_OWNERSHIP_UNVERIFIABLE` /
  `NATIVE_OWNERSHIP_RUN_MISSING`) без тихого fallback на JSON и без создания
  новой БД.
- **Один писатель.** Выбор и мутация сериализуются тем же межпроцессным run
  lock: его удерживает и native-исполнитель, и весь изменяемый legacy-хвост (от
  повторной проверки владельца до конца legacy-генерации), поэтому native и legacy
  писатели одного `run_root` не чередуются; занятый lock даёт exit `30`
  `NATIVE_RUN_LOCKED`. Нативный маршрут создаёт каталоги вывода и logger только
  после взятия lock; сбой создания каталога — exit `50`
  `NATIVE_OUTPUT_UNAVAILABLE`.
- **Провайдер лениво.** Ключ и провайдер строятся только для нового
  unattempted POST или для GET-only добора известного Media ID. Локальная
  пересборка из сохранённого raw receipt и починка экспортов завершённого
  прогона не читают ключ и не создают провайдера.
- **Порядок оплаты наследуется.** Резервация части → единственный POST →
  (для async `/media`: атомарная запись accepted `remote_task_id` → наблюдённая
  точная стоимость; для нативного sync `polza-tts` `/audio/speech`: приватный
  bounded HTTP body и receipt **до** status/JSON/audio parse) → атомарная запись
  оплаченных decoded raw-байтов и bounded receipt в `raw/` → связывание с БД →
  локальная конвертация/trim → CAS-коммит converted
  chunk → сборка из **упорядоченных DB-частей** (не по glob) → CAS-коммит
  завершения прогона. Ни один прежний маркер не даёт второй POST: `submitting`
  без подтверждения блокирует (`PAID_SUBMIT_UNCONFIRMED`), известный Media ID
  доводится только GET-запросами, валидный raw receipt пересобирается локально.
- **Синхронный маршрут без remote id.** Синхронный `polza-tts`
  (`/audio/speech`), `openrouter-tts` и `polza-chat-audio` (один streaming
  `POST /chat/completions` с инлайн-аудио) возвращают аудио инлайн и никогда не дают
  восстановимого task id. Их принятые decoded raw-байты и bounded receipt
  связываются одним CAS-переходом (`submitting` → `raw_saved`,
  `remote_task_id = null`). Точная наблюдённая стоимость `polza-tts` пишется в
  той же транзакции **до FFmpeg**; `openrouter-tts` и `polza-chat-audio` не
  сообщают синхронного usage, их стоимость остаётся неизвестной, не нулевой.
  Потерянный/неопределённый ответ оставляет `submitting`, блокирует resume и
  overwrite и никогда не разрешает второй POST/GET.

  Только нативный `polza-tts` `/audio/speech` сохраняет **до status check и
  parse** точное тело полученного HTTP-ответа в приватный
  `raw/<attempt_uuid>.response` (максимум 16 MiB, файл 0600, каталог 0700) и
  bounded `*.response.receipt.json`: UUID попытки/части, chunk number/id,
  synthesis fingerprint, provider/model/voice/**requested response format**,
  HTTP status, size/SHA-256 и очищенный opaque generation ID. Ключ, заголовок
  Authorization, текст запроса, тело или JSON keys ошибки не публикуются.
  HTTP-ошибка и malformed/неподдержанный audio сохраняются приватно, но не
  считаются успешным raw и не порождают retry. Ответ `3xx` не следует
  (`allow_redirects=False`) и обрабатывается как bounded HTTP status, так что
  редирект не превращается во второй платный POST. На `--resume`/`history resume`
  при `submitting` сначала проверяется **любая** часть response-evidence: только
  полный совпавший 2xx receipt+body разбирается локально с Decimal cost,
  сверяется с существующим decoded raw receipt при его наличии и сохраняет
  наблюдённую стоимость до FFmpeg. Отсутствующая половина, чужая identity,
  испорченный hash, неверный формат/тело или raw-конфликт дают
  `PAID_SUBMIT_UNCONFIRMED` без ключа, POST/GET или усвоения чужих байтов.
  Если response-evidence **вообще не существует**, прежний decoded raw receipt
  может восстановить старую попытку локально с неизвестной стоимостью:
  её нельзя придумать из аудиофайла. Для legacy/OpenRouter эта новая pre-parse
  гарантия не заявляется; Gemini 3.8 через Polza остаётся нестабильным и
  отправляется только по явному
  `--allow-experimental-gemini-speech-parts` (см. раздел S06), live не подтверждён.
- **Экспорт — проекция.** `run_state.json`, `chunks.json`, run/manifest JSON из
  одного verified DB-вида; каждый файл несёт `history_run_uuid` и
  `history_revision`, пишется атомарно и не даёт legacy JSON-разрешения на
  resume (`native_history` без `pending_attempt`). Повторный экспорт не делает
  провайдерских и платных действий. Сбой экспорта (в том числе ошибка
  проекции DB-вида) оставляет БД, raw и аудио нетронутыми и возвращает exit
  `50` с `details.error_code = NATIVE_EXPORT_FAILED`.
- **Безопасность доказательств и артефактов.** Владелец читается
  read-only-соединением, согласованным с активным WAL, поэтому только что
  закоммиченный прогон не теряется; legacy-ветка повторно проверяет владельца
  перед записью и удалением, а свежий нативный прогон отказывается занять
  каталог с чужим локальным состоянием (fail-closed, exit `30`
  `NATIVE_OWNERSHIP_*`). Дубли попыток/артефактов на часть отклоняются до
  любого нового POST/GET (`NATIVE_EVIDENCE_INCONSISTENT`); символьная ссылка
  дескриптора отклоняется без чтения цели, а convert/trim/concat пишут только
  в уникальный приватный staging-файл в том же каталоге и публикуются
  атомарно, не следуя symlink и не затирая внешний файл.
- **Ограничение на `--overwrite`.** Для нативного прогона `--overwrite`
  отклоняется (exit `30`, `details.error_code` = `NATIVE_OVERWRITE_UNSUPPORTED`),
  чтобы не удалить принятые платные доказательства; для явно нового запуска
  нужен другой `--run-id`. `--skip-existing` сохраняет приоритет. Новые
  опции, которые срез не поддерживает, на нативном прогоне отклоняются до
  провайдера (`NATIVE_OPTIONS_UNSUPPORTED`).
- **Локальные тайминги после готового аудио.** Когда прогон запросил
  `--with-timings --timing-provider faster-whisper`, эффективные настройки
  таймингов пишутся в снимок вместе с output-опциями, поэтому resume
  доказывает, что они не менялись (иначе exit `30`
  `NATIVE_PROCESSING_UNSUPPORTED` до провайдера). Перед **любым** новым
  платным POST проверяется доступность локальной модели faster-whisper без
  скачивания; недоступная зависимость или некэшированная модель — fail-closed
  (exit `10` `NATIVE_TIMING_MODEL_UNAVAILABLE`) до POST, и сам прогон всегда
  идёт с `local_files_only`, без неявного скачивания. После CAS-коммита
  готового аудио исполнитель пишет `<prefix>.timings.json` и `<prefix>.srt`
  через тот же приватный staging и связывает отдельный history-прогон
  `timings` через `parent_uuid` (не дублирует TTS-попытки, не запускает legacy
  finalizer/JSON-writer). Сбой извлечения — exit `50` `NATIVE_TIMING_FAILED`;
  сбой записи связанной истории — exit `50` `NATIVE_TIMING_HISTORY_FAILED`;
  оба сохраняют завершённое TTS-аудио, оплаченный raw и стоимость нетронутыми
  и удаляют только что записанные timing-файлы, чтобы экспорт не ссылался на
  несвязанные тайминги. Ссылки на тайминги попадают в экспорт/`files`
  (`timings_json`, `srt`) только при наличии verified связанного прогона.
- **Платные облачные тайминги после готового аудио.** Интегрированный шаг
  `--with-timings --timing-provider groq-whisper|xai-stt` проходит ту же стадию
  через тот же платный boundary, что и standalone `timings`: до POST пишется
  детерминированный приватный child-каталог (`.paid-timing`) с `submitting`-
  попыткой, exact raw-ответ сохраняется до fallible parse, а `history resume`
  воспроизводит/читает его локально без второго POST; неопределённый исход остаётся
  `PAID_SUBMIT_UNCONFIRMED` и блокирует retry/fallback/resume/overwrite. `history
  sync` никогда не отправляет запрос и сообщает pending тайминги неполными.
  Стоимость такого child-прогона всегда `NULL` (unknown), а родительский
  TTS-raw/cost остаются нетронутыми. Таблицы и schema не меняются.
- **Записанная семантика обрезки.** `--no-trim` — часть нативного маршрута, а
  не повод уйти в legacy JSON: значение записывается в output-опции снимка
  (`trim_final_silence`), поэтому при обрезке по умолчанию каждый сконвертированный
  фрагмент обрезается, а при `--no-trim` провайдерская тишина сохраняется. Resume
  обязан повторить то же значение: расхождение даёт exit `30`
  `NATIVE_PROCESSING_UNSUPPORTED` до провайдера, поэтому смена флага не может
  молча изменить уже принятое аудио.
- **Локальная проверка качества после готового аудио.** Когда запуск запросил
  установленный локальный `--tts-quality-provider` (`qwen-local` или
  `nemotron-local`), эффективные настройки проверки пишутся в выходные опции
  снимка, поэтому resume доказывает, что они не менялись (иначе exit `30`
  `NATIVE_PROCESSING_UNSUPPORTED` до провайдера). Перед **любым** новым платным
  POST проверяется наличие установленной локальной модели без скачивания;
  недоступная зависимость или отсутствующие локальные ассеты — fail-closed
  (exit `10` `NATIVE_QUALITY_MODEL_UNAVAILABLE`) до POST. Проверка идёт после
  CAS-коммита готового аудио, под тем же run lock, с существующей семантикой
  `verify-tts` (те же пороги similarity/word-ratio против собственного
  записанного текста сценария) и без legacy finalizer/JSON-writer. Наблюдённый
  транскрипт хранится приватно как текст-источник `verification_transcript`
  связанного прогона `verify` (`parent_uuid`), а `config_snapshot` того же
  прогона — контентно-пустой receipt (идентичность ASR, вердикт, similarity,
  причины) без сохранения ожидаемого текста; транскрипт не попадает ни в
  машинный вывод, ни в экспорты. Несовпадение сохраняет завершённое аудио,
  оплаченный raw и стоимость и **записывает наблюдённый FAIL** тем же связанным
  прогоном `verify` (`quality_passed: false`, similarity и фиксированные failure
  reasons, без ожидаемого текста) перед возвратом существующего quality exit `60`
  (`NATIVE_QUALITY_FAILED`); сбой локальной транскрипции — exit `50`
  `NATIVE_QUALITY_ASR_FAILED`; сбой записи связанной истории после наблюдённого
  вердикта — exit `50` `NATIVE_QUALITY_HISTORY_FAILED`, и прогон не заявляет
  сохранённый FAIL. `generate --resume`/`history resume UUID`
  восстанавливают запрошенные настройки из снимка без исходного скрипта и
  выполняют только недостающую локальную проверку; завершённый прогон не
  повторяет ASR и не дублирует связанный прогон, а записанный FAIL повторно
  сообщается тем же exit `60` без нового POST/ASR. `history sync UUID` не запускает
  ASR и не делает POST и сообщает записанный вердикт
  (`quality.complete: true, passed: false`) либо `quality.complete: false`, пока
  проверка не выполнена.

Standalone `verify-tts`, `transcribe` и локальный/облачный `timings` также
записывают свой результат в canonical SQLite (см. раздел «Локальные ASR /
timings / verify в канонической истории»). Локальные семейства
`qwen-local`/`omnivoice-local` дополнительно допускают `format: voiceover`
для фактически работающих сочетаний: любой режим `qwen-local` (preset сохраняет
разрешённый voiceover-валидатором голос, clone/design подменяют его своим
маркером режима, как и в legacy) и preset-ветку OmniVoice-банка. Голоса
voiceover-сценария по-прежнему валидируются до маршрута: `format: voiceover` с
`openrouter-tts` и style prompt, с недопустимым для provider/model голосом, с
OmniVoice-режимом, который отклоняет `--voice`, и с OmniVoice-банком, у которого
нет профиля с разрешённым маркерным id, остаётся usage error (exit `2`)
до любого провайдера. Локальный faster-whisper timing и локальная проверка
качества интегрированы в
нативный прогон (см. выше). Оба нативных диалоговых маршрута описаны в
подразделах ниже.
Пользовательские глаголы
`history resume ID` / `history sync ID`, которые восстанавливают такой прогон из
снимка, описаны в разделе ниже.

### Нативный OpenRouter Gemini dialogue

Первый диалоговый маршрут, переведённый на canonical SQLite —
`--provider openrouter-tts --model google/gemini-3.1-flash-tts-preview
--format dialogue` (валидированный YAML, ровно два speaker'а с различными
голосами из `GEMINI_TTS_VOICES`) с обязательным установленным локальным
`--tts-quality-provider` (`qwen-local`/`nemotron-local`) или платным облачным
`xai-stt`. Записанная семантика
обрезки (`--no-trim`) и интегрированные
`--with-timings` (локальный `faster-whisper` или платные `groq-whisper`/`xai-stt`)
входят в тот же маршрут; любой `polza-tts` dialogue и `openrouter-whisper`
остаются на legacy executor, новый обобщённый движок не добавляется.

- Каждая реплика — один paid-запрос со своим cast-voice (`voice`, как в legacy);
  снимок хранит порядок реплик, speaker, cast/effective voice, паузу и текст.
- Перед concat каждая реплика проходит обязательную ASR-проверку. Локальная
  модель (`qwen-local`/`nemotron-local`) прощупывается до первого paid POST (без
  implicit download); платный облачный `xai-stt` проверяет наличие ключа до
  первого paid TTS POST, а каждая реплика получает свой приватный child-root
  `.paid-quality/<turn>` и `submitting`-попытку до своего POST. Наблюдённый
  вердикт PASS/FAIL и приватный
  `verification_transcript` сохраняются на самом TTS-прогоне и связаны с этой
  частью; провал проверки — exit `60` (`NATIVE_QUALITY_FAILED`), сбой ASR —
  exit `50` (`NATIVE_QUALITY_ASR_FAILED`), сбой записи — exit `50`
  (`NATIVE_QUALITY_HISTORY_FAILED`), отсутствующая локальная модель — exit `10`
  (`NATIVE_QUALITY_MODEL_UNAVAILABLE`), отсутствующий ключ — exit `20`
  (`NATIVE_PAID_QUALITY_KEY_MISSING`).
- Записанный FAIL не повторяет POST и не перезапускает модель: `generate
  --resume`, `history resume ID` и `history sync ID` повторно сообщают exit `60`,
  сохраняя оплаченные raw/аудио/стоимость. Неопределённый синхронный submit
  остаётся `PAID_SUBMIT_UNCONFIRMED` и блокирует retry и `--overwrite`.
- Сборка использует существующий `concat_dialogue_turns` с записанным планом пауз
  `250`/`600`/`0` мс и порядком частей; совместимые JSON-экспорты строятся из БД
  и несут `history_run_uuid`/`history_revision`, а также per-turn
  `turn_index`/`speaker`/`voice`/`pause_after_ms`/`audio_sha256` и
  контентно-пустой `tts_quality` receipt (без приватного транскрипта).

### Нативный локальный OmniVoice dialogue

Второй диалоговый маршрут — `--provider omnivoice-local --model
audio-cpp/omnivoice-q8_0 --mode preset --voice-bank <catalog.json> --format
dialogue` (валидированный YAML, ровно два speaker'а с различными profile ID из
admitted bank и различными `reference_sha256`). Записанная семантика обрезки
(`--no-trim`), интегрированные `--with-timings` (локальный `faster-whisper` или
платные `groq-whisper`/`xai-stt`) и установленный локальный **или** платный
облачный `xai-stt` `--tts-quality-provider`
входят в тот же маршрут; незарегистрированный quality-провайдер, другие режимы
OmniVoice и любой `polza-tts` dialogue остаются на legacy executor.

- Каждая реплика — один локальный запрос со своим cast voice-bank profile;
  снимок хранит порядок, speaker, cast/effective voice, reference digest,
  паузу и текст, а также локатор каталога и настройки выбранных профилей.
- Платной попытки и стоимости нет: каждая реальная локальная реплика пишет свою
  отдельную durable-попытку `local_tts_chunk` (`status`/`cost` NULL, никогда не
  платный маркер), а `history costs` считает такие строки в
  `local_attempts_without_api_charge`, не как облачную неизвестную стоимость и не
  как выдуманный ноль. Raw-байты реплики и её converted chunk линкуются к этой
  попытке до конвертации, поэтому сбой FFmpeg восстанавливается из raw без
  повторного запуска модели; упавший или прерванный локальный запуск сохраняет
  свой truthful outcome (`local_failed` или pending-строка) и безопасно
  повторяется новой попыткой при явном `--resume`/`history resume`.
- Опциональная ASR-проверка: когда прогон запросил установленный
  локальный `--tts-quality-provider` (`qwen-local`/`nemotron-local`) или платный
  облачный `xai-stt`, каждая реплика проходит ту же строгую
  сверку до concat (для облачного провайдера — через тот же платный boundary и
  свой child-root на реплику), а наблюдённый вердикт PASS/FAIL и приватный
  `verification_transcript` связываются с частью; записанный FAIL повторно
  сообщается exit `60` без нового запуска модели/POST, а без провайдера сохраняется
  прежнее поведение без проверки.
- Если reference-файл профиля пропал или его digest изменился, запуск
  завершается ошибкой (`NATIVE_LOCAL_REFERENCE_UNAVAILABLE` или
  `NATIVE_RESUME_IDENTITY_CHANGED`) до вызова локальной модели.
- `history sync ID` не запускает локальную модель: незавершённая локальная часть
  сообщает `NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED`, завершённый прогон только
  переписывает совместимые JSON-экспорты из БД. Реальный локальный запуск модели
  в тестах — `NOT_RUN`.

### Нативный локальный OmniVoice

Нативные не-диалоговые маршруты `--provider omnivoice-local --model
audio-cpp/omnivoice-q8_0` (plain Markdown без frontmatter, а для preset-ветки
банка — ещё и `format: voiceover`):

- `--mode preset --voice-bank <catalog.json>`: весь script сливается в одну
  OmniVoice session (один локальный вызов), которая клонирует выбранный profile
  каталога.
- `--mode auto`, `--mode clone --reference-audio <файл> --reference-text <текст>`
  и `--mode design --design-instruction <инструкция>`: те же один session-вызов и
  существующие local attempt/raw seams. `auto` отправляет в runtime только
  `omnivoice_mode=auto` без style/reference/design полей — голос выбирает
  upstream-модель (`voice_selection.kind = auto-voice`), никакой preset не
  подменяется; записанный effective voice прогона — только маркер режима
  (`auto`, `clone`, `design`). Длинный design по-прежнему
  отклоняется существующей проверкой длины до admission, а короткий русский
  design остаётся experimental.

Записанная семантика обрезки (`--no-trim`), интегрированные
`--with-timings` (локальный `faster-whisper` или платные `groq-whisper`/`xai-stt`)
и установленный локальный
`--tts-quality-provider` (`qwen-local`/`nemotron-local`) входят в тот же
маршрут; облачный quality-провайдер остаётся на legacy executor.
`--mode preset --voice-bank` допускается и для `format: voiceover`:
валидатор такого сценария разрешает единственный голос
`built-in-female-style-condition`, поэтому admitted-каталог обязан содержать
профиль с этим id, а прогон коммитит именно его identity. Остальные
OmniVoice-режимы всегда отклоняют `--voice`, который voiceover поставляет, так
что `format: voiceover` с `auto`/`clone`/`design` остаётся usage error.

- Снимок хранит выбранную идентичность: для preset — локатор каталога, mode,
  локатор/SHA/text/language выбранного профиля, model и effective voice (= profile
  id); для `auto`/`clone`/`design` — свой блок `omnivoice_mode` (mode, model,
  effective voice, а для clone — канонический локатор референса с SHA-256 и
  размером и точный reference text, для design — точная инструкция; сами байты
  референса и инструкция в публичные JSON не попадают). Публичные JSON-экспорты
  публикуют provider/model/voice (profile id или маркер режима) и текст, но не
  reference text каталога и не design-инструкцию.
- Платной попытки нет: единственный локальный вызов пишет свою durable-попытку
`local_tts_chunk` (`status`/`cost` NULL, никогда не платный маркер), а raw-байты
линкуются к попытке до конвертации, поэтому сбой FFmpeg восстанавливается из raw
без повторного запуска модели; `history costs` считает эту попытку в
`local_attempts_without_api_charge`, а не как облачную неизвестную стоимость и не
как выдуманный ноль.
- Пропавший reference или изменившийся digest даёт
`NATIVE_LOCAL_REFERENCE_UNAVAILABLE` / `NATIVE_RESUME_IDENTITY_CHANGED` до вызова
локальной модели. Для non-preset режимов тот же fail-closed порядок применяется
к `omnivoice_mode`-блоку: изменившиеся reference-байты, инструкция или mode
дают `NATIVE_RESUME_IDENTITY_CHANGED` до провайдера на `generate --resume`, а на
`history resume` — проверка коммитнутого reference до модели. `history sync ID`
локальную модель не запускает: незавершённая
локальная часть сообщает `NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED`, завершённый
прогон только переписывает совместимые JSON-экспорты из БД.
- Реальный локальный запуск модели (audio.cpp/OmniVoice) в тестах — `NOT_RUN`.

### Нативный локальный Qwen

Третье нативное локальное семейство маршрутов — обычные не-диалоговые режимы
`qwen-local`: клон-маршрут
`--provider qwen-local --mode clone --sample <файл> [--sample-text <текст>]` и
инструктированные маршруты `--mode preset` (модель CustomVoice, `--voice` из
каталога `QWEN_PRESET_SPEAKERS`) и `--mode design` (модель VoiceDesign,
обязательный непустой `--qwen-instruct`). Эти же режимы допускаются и для
`format: voiceover`: валидатор разрешает один голос на весь сценарий, затем
`preset` сохраняет его как effective voice, а `clone`/`design` подменяют его
своим маркером режима (`clone`/`design`) точно так же, как legacy executor, —
голос из frontmatter для этих двух режимов в legacy игнорируется, и новая
семантика не вводится. Режим `auto` не реализован и
отклоняется как usage error (exit `2`) до провайдера, модели и снимка:
автоматический выбор режима не подменяется на `preset`; неизвестное
значение `VOICEOVER_QWEN_TTS_RUNTIME` остаётся на legacy executor. Записанная
семантика обрезки (`--no-trim`), интегрированные
`--with-timings` (локальный `faster-whisper` или платные `groq-whisper`/`xai-stt`)
и установленный локальный
`--tts-quality-provider` (`qwen-local`/`nemotron-local`) входят в тот же
маршрут; облачный quality-провайдер остаётся на legacy executor.

- Снимок хранит полную неизменяемую идентичность клона: канонический абсолютный
  локатор референс-файла, его SHA-256 и размер (сами байты в снимок и публичные
  JSON не копируются), точный `reference text`, а также выбранную модель, режим
  (`clone`) и рантайм/язык. Для `preset`/`design` снимок хранит свой блок
  идентичности (`mode`, `model`, `voice`, точная инструкция, рантайм и язык),
  а публичный `style_prompt` остаётся прежним полем. Любое расхождение блокирует
  resume до вызова модели.
- Платной попытки и стоимости нет: каждая реальная локальная часть пишет свою
  durable-попытку `local_tts_chunk` (`status`/`cost` NULL, никогда не платный
  маркер), raw-байты и converted chunk линкуются к этой попытке до конвертации,
  поэтому сбой FFmpeg восстанавливается из raw без повторного запуска модели, а
  упавший/прерванный запуск сохраняет truthful outcome и безопасно повторяется
  новой попыткой при явном `--resume`/`history resume`. `history costs` считает
  эти попытки в `local_attempts_without_api_charge`, а не как облачную
  неизвестную стоимость и не как выдуманный ноль.
- Перед первой локальной синтез-частью (и до резервации попытки) проверяются
  установленный локальный рантайм с кэшированной моделью без скачивания; для
  клона — дополнительно референс-файл (наличие, размер, SHA-256): пропавший или
  изменённый референс и смена рантайма дают
  `NATIVE_LOCAL_REFERENCE_UNAVAILABLE` (exit `30`), недоступная зависимость или
  некэшированная модель — `NATIVE_LOCAL_MODEL_UNAVAILABLE` (exit `10`), до
  вызова модели. На `generate --resume` изменённый референс даёт
  `NATIVE_RESUME_IDENTITY_CHANGED`, а отсутствующий `--sample` — usage error
  (exit `2`).
- `history resume ID` реконструирует прогон из снимка без исходного сценария и
  перед моделью проверяет тот же референс; `history sync ID` модель не запускает и
  незавершённую часть сообщает `NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED`, а
  завершённый прогон только переписывает совместимые JSON-экспорты из БД. Реальный
  локальный запуск модели в тестах — `NOT_RUN`.

## Локальные ASR / timings / verify в канонической истории

Локальные маршруты `transcribe` (`qwen-local`, `nemotron-local`), `timings` с
`--asr-provider` и локальный `--timing-provider faster-whisper`, а также
`verify-tts` записывают наблюдаемый результат в каноническую SQLite-историю по
умолчанию. Так же сохраняется standalone облачный
`timings --timing-provider groq-whisper|xai-stt` через платный boundary (см.
ниже), а облачный dialogue-QA `xai-stt` допущен на двух диалоговых маршрутах
через тот же boundary. Вне канонической истории остаются только
`openrouter-whisper` (не даёт реальных таймстемпов и отклоняется) и будущий
облачный ASR-маршрут до подтверждённого paid-submit контракта (S01/S07).

- Управление — `settings.toml` в текущем каталоге: `[history] enabled = false`
  отключает запись полностью, и команда работает как раньше, не создавая дом,
  БД, `runs/` и sidecar. Отсутствие файла, секции или ключа означает
  `enabled = true`. Нечитаемый `settings.toml` или не-boolean `enabled`
  fail-closed: результат отдаётся, но помечается сбоем сохранения.
- Операции: `asr` (`transcribe`), `timings` (локальный `timings`),
  `verify` (`verify-tts`). `history list --operation asr|timings|verify`
  фильтрует их, `history show ID` рендерит попытку, артефакты и текстовые роли.
- `run_root` — приватный `0700` каталог managed `VOICEOVER_HOME/runs/<uuid>`, а
  не пользовательский `out/<run-id>`: ASR-прогон не делит каталог с
  legacy/native TTS-прогоном и не может быть удалён его `--overwrite`.
- Исходное аудио не копируется: сохраняются внешний абсолютный путь, размер и
  SHA-256. Если файл исчез до записи, артефакт получает
  `availability: "missing"` без выдуманного размера/хеша, а уже сохранённый
  transcript остаётся валидным.
- Попытка одна, `status: "completed"`, цена `null` (неизвестная): локальные
  маршруты не имеют внешнего API-начисления, и `history costs` считает такую
  попытку в `local_attempts_without_api_charge`.
- Приватные роли текста: `asr_transcript` (распознанная речь), `asr_context`
  (переданный `--context`/`--context-file` prompt), `verification_transcript`
  (фактическая проверка). `verify-tts` **никогда** не сохраняет ожидаемый
  TTS-текст и не перезаписывает исходный сценарий; при совпадении audio с
  native TTS-run сохраняется связь `parent_uuid`.
- Наблюдаемые сегменты и происхождение попадают в приватный
  `config_snapshot` (модель, runtime, `timestamp_mode`/`timestamp_basis`,
  сегменты с их собственными `null`-границами). Выдуманные spans не создаются,
  text-only результат не становится SRT, а пустой transcript сохраняется как
  `""` с `text_completeness: "incomplete"`.
- Файловые артефакты ссылаются только на реально существующие файлы:
  `timings_json`/`srt` появляются в истории лишь когда команда их записала.
- `--json` при успешном сохранении сохраняет прежние поля и exit code и
  добавляет безопасный блок `"history": {"saved": true, "run_uuid": "..."}`.
  Если запись в БД не удалась, команда **не** заявляет о сохранении: она
  отдаёт уже полученный результат (transcript, файлы, receipt) с
  `"status": "partial"` и
  `"history": {"saved": false, "error_code": "HISTORY_PERSISTENCE_FAILED"}`,
  завершается exit `50` и не вызывает провайдера повторно. `verify-tts` в этом
  случае тоже возвращает `50` вместо `0`/`60`, а качество остаётся видно в
  `receipt.status`.
- Приватность: `history.sqlite3` — plaintext-БД, не зашифрованный сейф.
  `list`/`show`/`costs` показывают только метаданные: роль текста, язык,
  `has_content` и хеш, но не сам transcript, не context prompt и не подписанные
  URL.

```bash
voiceover transcribe --audio recording.wav --provider qwen-local --json
voiceover timings --audio recording.wav --asr-provider qwen-local --output-dir out --run-id rec-timings --json
voiceover verify-tts --audio out/prod/full.mp3 --expected-file script.md --provider qwen-local --json
voiceover history list --operation asr --json
voiceover history show <run-uuid> --json
```

### Облачный `timings` (`groq-whisper` / `xai-stt`) — платный boundary

Standalone `voiceover timings --timing-provider groq-whisper|xai-stt` теперь
сохраняет прогон в канонической истории и защищает единственный платный POST:

- **Маркер до POST.** Сначала валидируется читаемый regular source
  (канонический путь, размер, SHA-256), затем коммитится строка run с identity
  запроса (provider, model, language, timestamp granularity, формат) и попытка
  `status="submitting"`; `run_root` — это канонический output-root прогона, а не
  сгенерированный history UUID. Провайдер вызывается только после успешного
  коммита; сбой коммита означает **ноль** запросов.
- **Один владелец на output-root.** Свежий вызов `timings` или `--overwrite`
  против уже принадлежащего платному прогону root (включая `submitting` и
  `completed`) fail-closed завершается `PAID_TIMING_OUTPUT_OWNED` **до**
  удаления, чтения ключа и POST: владение привязано к каноническому output-root
  (exact, untruncated lookup по `run_root` + платному origin, а не скан newest-N,
  плюс приватный descriptor `.voiceover-paid-timing.json`). Отсутствующий
  descriptor уступает committed-строке, а нечитаемая/чужая/новая/неоднозначная
  БД завершается fail-closed, а не считается unowned. Все writer'ы делят одно
  пространство канонического root, поэтому guard двусторонний под тем же run
  lock: `generate` (native и legacy) отклоняет paid-owned root и его
  descriptor/committed-строку/raw-evidence до выбора, удаления, чтения ключа и
  POST, а маршрут `timings` так же отклоняет native-history-owned root. Явно
  другой `--run-id`/output-dir — отдельное платное решение; `history
  resume`/`sync` работают по UUID этого прогона. Один cross-process run lock на
  output-root удерживается через ownership, reservation, POST, persistence и
  публикацию.
- **Сырой ответ до парсинга.** Сначала приватный receipt, затем точное тело
  успешного ответа пишутся в `raw/<attempt>.body` (оба fsync/atomic) и только
  потом линкуется артефакт `provider_raw_response` (`managed_relative`,
  sha256/size) и попытка переходит в `raw_saved` — одной DB-транзакцией. Если
  запись/линковка не удалась, попытка остаётся `submitting`, parse не выполняется.
- **Crash-reconciliation.** Receipt публикуется **до** тела, поэтому crash сразу
  после замены тела оставляет проверяемую пару; явный `history resume`
  восстанавливает ту же попытку (`submitting → raw_saved`) и локально
  воспроизводит тело (**0 POST**). Reconciliation принимает ровно нуль или один
  совпадающий stored raw-артефакт, а каждый replay проверяет тот же артефакт
  (attempt, managed path, обязательные size/hash) против receipt/тела, а не
  выбирает первую строку; `history sync` никогда не reconcile-ит, а
  mismatch/missing/tampered/oversized/symlink/дубликат/чужой receipt или тело
  остаются заблокированы.
- **Завершение.** После успеха записываются transcript (`asr_transcript`), ссылки
  на `<prefix>.timings.json`/`.srt` (публикуются atomic temp-replace внутри
  проверенного output-root) и provenance (`timestamp_basis`:
  `provider_segment_timestamps`, `derived_from_provider_words` или
  `fallback_full_text`), попытка переходит в `completed`, run закрывается одной
  транзакцией; пропавший обязательный артефакт завершает прогон fail-closed.
- **Источник и путь.** Продолжение проверяет источник (path/size/SHA) **до**
  parse/ffprobe/публикации; замена/удаление источника отказывает в recovery и
  сохраняет raw. Symlink-лист артефакта не разворачивается, source внутри
  удаляемого output-subtree запрещает удаление.
- **Безопасность при неопределённости.** Сбой/таймаут POST оставляет попытку
  `submitting` — никакой автоматический retry, resume или sync не сделает второй
  POST. `history resume`/`sync` читают состояние только через read-only
  consistent view (без migrate, journal-switch и sidecar): `-wal` без пригодного
  `-shm` отклоняется, а не открывается (чтение не создаёт sidecar), и
  run/attempt/artifact-выборки идут в одной явной deferred read-транзакции,
  поэтому вклинившийся commit writer'а не рвёт снимок; повреждённая или
  несовместимая БД — fixed error, а не «unknown UUID», и нечитаемая/непригодная
  БД не трактуется как unowned перед удалением.
- **Деньги и приватность.** Цена всегда `null` (unknown), remote id не
  выдумывается. Сырое тело и transcript приватны и не попадают в
  `--json`/`history show`/`costs`; `history costs` видит облачную попытку как
  unknown (`completeness: partial`), не как локальную.
- **Требуется history.** Если `settings.toml` отключает историю, команда
  fail-closed завершается с `PAID_TIMING_HISTORY_REQUIRED` **до** запроса,
  удаления и чтения ключа.
- **Ключ до удаления и резервации.** Под уже взятым output-root lock сначала
  обрабатываются credential-free случаи: существующий каталог без `--overwrite`
  (exit `30`) и `--skip-existing` (exit `0`, `status: skipped`) — ключ для них
  не читается. Затем резолвится ключ выбранного провайдера (`GROQ_API_KEY` для
  `groq-whisper`, `X_AI_API_KEY` для `xai-stt`), и только после этого
  `--overwrite` удаляет каталог и резервируется платная попытка. Непригодный
  явный `--env-file` или отсутствующий ключ дают exit `20` с тем же redacted
  сообщением: каталог не удаляется, ложный `submitting`-маркер не пишется,
  провайдер не вызывается. Если явный env-файл сломался уже после preflight, та
  же фиксированная credential-ошибка не маскируется под ASR failure.

```bash
voiceover timings --audio recording.wav --timing-provider groq-whisper --output-dir out --run-id rec --json
voiceover history resume <run-uuid> --json   # локальный replay без второго POST
voiceover history sync <run-uuid> --json     # только чтение, модель не запускается
```

**Интегрированное переиспользование.** Тот же boundary обслуживает два
фактически работающих облачных post-audio маршрута:

- `generate --with-timings --timing-provider groq-whisper|xai-stt` на
  допустимом нативном TTS/dialogue: платёжный timing-child коммитится в
  детерминированный приватный каталог `<run-root>/.paid-timing` (родительский
  native-root не перезаписывается), raw сохраняется до парсинга, тайминги
  публикуются в родительский root, а `history resume`/`sync` читают его локально.
- `--tts-quality-provider xai-stt` только на двух диалоговых маршрутах
  (OpenRouter Gemini и OmniVoice preset-банк), по реплике: каждая реплика
  получает свой child-root `<run-root>/.paid-quality/<turn>` до POST; вердикт
  PASS/FAIL и приватный transcript линкуются к родительской части (тот же per-turn
  механизм, что и для локальной проверки), а записанный FAIL повторно сообщается
  без нового POST. Не-диалоговый `xai-stt` по-прежнему игнорируется legacy
  executor'ом.

Стоимость каждого child-прогона `NULL` (unknown); ни один маршрут не создаёт
новую схему или таблицу.

## `verify-tts` — fail-closed quality gate

Команда запускает явно выбранный зарегистрированный ASR route и сравнивает
нормализованный transcript с `--expected-text` или `--expected-file`. Она
проверяет similarity, существенные пропуски, неожиданные слова и повторённые
двух-/трёхсловные последовательности. Transcript и expected text не публикуются:
stdout/`--receipt` содержат только hashes, counts, ASR identity и audio SHA-256.
`0` означает technical PASS, `60` — quality FAIL. Недоступный ASR остаётся
exit `10`/`30`; decode-only PASS не подменяет эту проверку. Даже technical PASS
содержит `human_listening_required: true`. Успешный прогон также сохраняет
приватный `verification_transcript` в каноническую историю (см. раздел
«Локальные ASR / timings / verify в канонической истории»); ожидаемый текст не
сохраняется.

```bash
voiceover verify-tts --audio out/run/full.mp3 --expected-file script.md \
  --provider qwen-local --model Qwen/Qwen3-ASR-0.6B --language ru \
  --receipt out/run/tts-quality.json --json
```

Named `--voice` (вне preset+bank), Qwen cloning/sample options и style
controls для этого provider fail closed. На Windows Docker/WSL не выбираются:
нужен `VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE` с рядом лежащим
`audio_cpp_dependency_closure.json`, проверяющим SHA-256 EXE/DLL closure, и
`VOICEOVER_OMNIVOICE_MODEL` и тот же noncommercial-use acknowledgment.
Отсутствующий, несовпавший SHA-256 или closure даёт unavailable route;
fallback к container нет. Полный pinned receipt и ограничения
лицензии: [OmniVoice Local TTS](omnivoice-local-tts.md).

## `history` — локальная история прогонов (S04)

S04 добавляет локальную SQLite-историю. `history list`, `history show` и
`history import` читают и импортируют метаданные независимо от текущего рабочего
каталога. Сами `list`/`show`/`import` не делают
провайдерских, ASR, сетевых или платных вызовов; генерацию на canonical writer
переключают native-маршруты (см. разделы выше). Локальные `transcribe`,
`timings` и `verify-tts` добавляют прогоны операций `asr`, `timings` и `verify`
через тот же общий слой (см. раздел «Локальные ASR / timings / verify в
канонической истории»).

```bash
voiceover history list [--label L] [--operation O] [--status S] [--limit N] [--offset N] --json
voiceover history show ID --json
voiceover history import DIR --dry-run --json
voiceover history import DIR --json
voiceover history costs --json
```

- `--json` ставится в конце конкретной leaf-команды, не глобально. Как и везде,
  при `--json` stdout содержит ровно один JSON object, диагностика идёт в stderr,
  а exit code остаётся семантическим.
- `list`/`show`/`import` используют только существующие числовые коды `0`, `2`,
  `30`, `50`; стабильная машиночитаемая причина передаётся строкой
  `details.error_code`, без echo сохранённых строк БД.
- `--limit` — целое `1..500` (default `50`), `--offset` — неотрицательное целое.
  Неверное значение → exit `2` `HISTORY_INVALID_LIMIT` / `HISTORY_INVALID_OFFSET`.

### Дом истории и приватность

- Дом берётся из `VOICEOVER_HOME` и обязан быть абсолютным; относительное значение
  отклоняется exit `2` `HISTORY_HOME_INVALID` до любого чтения/записи. Без override
  используется системный data directory (`$XDG_DATA_HOME/voiceover-pipeline`,
  macOS `~/Library/Application Support/voiceover-pipeline`, Windows
  `%LOCALAPPDATA%\voiceover-pipeline`) — расположение не зависит от CWD.
- Управляемые каталоги создаются с приватным режимом `0700`. Существующий
  POSIX-каталог, доступный группе или миру (например `0755`), отклоняется
  (`HISTORY_HOME_PERMISSION`, exit `50`) вместо `chmod` чужого дерева и до записи
  plaintext-БД.
- `history.sqlite3` — обычная SQLite-БД и **не** зашифрованный сейф. Приватная БД
  намеренно хранит доступные тексты (сценарий, произносимый текст, инструкции,
  transcript); защита — права ОС, а не шифрование.
- Публичный вывод list/show остаётся content-free: полный сценарий, prepared text,
  transcript, raw snapshots, config snapshot и секрет-подобные значения
  (подписанные URL, Authorization) не печатаются. Присутствие текста/ошибки
  сообщается булевым полем (`has_content`, `has_error`), а не значением.

### Чтение без изменения канонической БД и sidecar

- Отсутствующая БД: `list` возвращает пустой список (`runs: []`, `count: 0`,
  `database.exists: false`), `show` — `HISTORY_RUN_NOT_FOUND` (exit `2`); ни дом,
  ни `runs/`, ни БД не создаются.
- `list`/`show`/`costs` и read-only preflight для `resume`/`sync` используют
  один проверенный consistent reader без миграций, DDL и WAL-switch. При
  отсутствии `-wal` это `mode=ro&immutable=1`: quiescent-дом не получает новую
  пустую пару `-wal`/`-shm`. При наличии `-wal` и уже существующего пригодного
  `-shm` применяется `mode=ro`: читаются закоммиченные WAL-кадры, в том числе
  после `index build`; устаревшая **пустая** пара тоже не блокирует `list/show`.
  Основной файл и `-wal` не переписываются, пара не удаляется; SQLite может
  обновить байты **существующего** `-shm` для read-lock/index bookkeeping.
  `history show` читает метаданные run/parts/attempts/artifacts/text sources в
  одной deferred read-транзакции: соседний commit не смешивает версии строк.
  Нельзя удалять sidecar вручную во время работы другого процесса.
- `-wal` без пригодного `-shm`, hot `-journal`, symlink и иностранная/повреждённая
  БД fail-closed (`HISTORY_DATABASE_UNREADABLE`, exit `30`); более новая схема
  и чужой migration ledger также отказываются с
  `HISTORY_DATABASE_TOO_NEW`/`HISTORY_DATABASE_CHECKSUM_MISMATCH` (exit `30`).
- `import --dry-run` ничего не пишет: не создаёт БД, каталоги, sidecar и не меняет
  оригиналы. При нечитаемой/WAL/иностранной БД dry-run возвращает exit `0`,
  помечает `database.readable: false` и добавляет `database_status_unknown` в
  `scan_conflicts`, а `already_imported` становится `null` (unknown), а не
  выдуманным `false`.

### `history list --json`

```json
{
  "status": "success",
  "dry_run": false,
  "database": {"path": "...", "exists": true, "schema_version": 3},
  "filters": {"label": null, "operation": null, "status": null},
  "limit": 50,
  "offset": 0,
  "count": 1,
  "runs": [
    {
      "run_uuid": "...",
      "operation": "tts",
      "user_label": "prod",
      "parent_uuid": null,
      "status": "completed",
      "run_root": "...",
      "legacy_source_root": "...",
      "record_version": 1,
      "created_at": "...",
      "updated_at": "...",
      "legacy_import": true
    }
  ]
}
```

`--label`/`--operation`/`--status` — точные фильтры, не подстроки. `config_snapshot`
не публикуется.

### `history show ID --json`

`ID` — внутренний `run_uuid` или точная пользовательская метка (`user_label`).
UUID-shaped значение сначала ищется как UUID, затем как метка.

- Не найдено → exit `2` `HISTORY_RUN_NOT_FOUND`.
- Метка совпала в нескольких roots → exit `2` `HISTORY_LABEL_AMBIGUOUS` с
  `candidate_count` и ограниченным списком `candidates`; первый кандидат не
  выбирается автоматически. Одинаковая метка в разных roots — разные UUID (§6).

```json
{
  "status": "success",
  "dry_run": false,
  "database": {"path": "...", "exists": true, "schema_version": 3},
  "run": {"run_uuid": "...", "operation": "tts", "user_label": "prod", "...": "..."},
  "parts": [
    {
      "part_uuid": "...",
      "position": 1,
      "voice": "alloy",
      "fingerprint": "...",
      "stage": "completed",
      "has_prepared_text": true,
      "has_shared_vibe": false,
      "has_specific_vibe": false,
      "has_effective_vibe": false
    }
  ],
  "attempts": [
    {
      "attempt_uuid": "...",
      "part_uuid": "...",
      "call_type": "tts_chunk",
      "provider": "polza-tts",
      "model": "...",
      "remote_id": "...",
      "status": "completed",
      "cost": {
        "amount": "0.1000",
        "currency": "RUB",
        "source": "legacy_import",
        "exact_available": false,
        "raw": "0.1000"
      },
      "has_usage": false,
      "has_error": false,
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "artifacts": [
    {
      "artifact_uuid": "...",
      "role": "chunk_audio",
      "path_kind": "external_absolute",
      "path": "...",
      "mime": "audio/mpeg",
      "size_bytes": 8,
      "sha256": "...",
      "availability": "present",
      "has_media_metadata": false
    }
  ],
  "text_sources": [
    {
      "text_source_uuid": "...",
      "part_uuid": "...",
      "artifact_uuid": null,
      "kind": "tts_script",
      "origin": "legacy_import",
      "content_hash": "...",
      "language": null,
      "text_completeness": "complete",
      "has_content": true
    }
  ]
}
```

Контракт денег (§5/§6): `amount` — десятичная строка; `"0"`/`"0.0"` — реальный
наблюдённый ноль, `null` — неизвестная цена. `exact_available: false` с
`source: "legacy_import"` означает сохранённый legacy-lexeme без гарантии
точности (`"0.1000"` сохраняется verbatim, `raw` держит исходный токен);
провайдерски подтверждённая exact-строка несёт `exact_available: true`.
Неизвестная цена остаётся отличима от нуля.

### `history import DIR --dry-run --json`

```json
{
  "status": "success",
  "dry_run": true,
  "source": "...",
  "database": {"path": "...", "exists": false, "readable": true},
  "discovered_count": 1,
  "importable_count": 1,
  "already_imported_count": 0,
  "missing_text_count": 0,
  "missing_audio_count": 0,
  "scan_conflicts": [],
  "runs": [
    {
      "source_root": "...",
      "run_id": "prod",
      "operation": "tts",
      "status": "completed",
      "provider": "polza-tts",
      "model": "...",
      "voice": "alloy",
      "chunk_count": 1,
      "chunks_with_text": 1,
      "chunks_missing_text": 0,
      "chunks_with_audio": 1,
      "chunks_missing_audio": 0,
      "chunks_with_cost": 1,
      "cost_total": "0.1000",
      "cost_total_exact": null,
      "cost_currency": "RUB",
      "script_text_available": true,
      "already_imported": false,
      "importable": true,
      "conflicts": []
    }
  ]
}
```

Dry-run сообщает найденные каталоги/записи, importable/уже импортировано,
отсутствующие тексты и аудио, конфликты. `already_imported` — `true`/`false`
только при читаемой БД, иначе `null`. Полный текст не печатается.

### `history import DIR --json` (реальный импорт)

```json
{
  "status": "success",
  "dry_run": false,
  "source": "...",
  "database": {"path": "..."},
  "imported_count": 1,
  "skipped_count": 0,
  "rejected_count": 0,
  "runs": [
    {
      "source_root": "...",
      "run_uuid": "...",
      "created": true,
      "parts": 1,
      "attempts": 1,
      "artifacts": 6,
      "text_sources": 2,
      "conflicts": []
    }
  ]
}
```

- Импорт полностью offline: без TTS/ASR/embeddings/сети/платных вызовов и без
  изменения оригинальных файлов.
- Каждый run пишется одной транзакцией; повторный импорт того же root идемпотентно
  пропускается (`imported_count: 0`, `skipped_count: 1`, тот же `run_uuid`) и не
  дублирует части, costs или text_sources. `rejected_count` — каталоги, чья
  идентичность не проходит fail-closed.
- Импорт в существующую приватную БД требует дом `0700`; доступный группе/миру дом
  отклоняется (`HISTORY_HOME_PERMISSION`, exit `50`) до создания plaintext-БД.
  Ошибка записи → exit `50` `HISTORY_WRITE_ERROR`; нечитаемая/более новая/
  иностранная БД → exit `30`.

### `history costs --json` — read-only учёт расходов

`history costs` — глобальный read-only отчёт по каноническому `attempts`:
итоги по валютам и разбивка по операции × валюте. Синтаксис минимальный — без
фильтров, только `--json` в конце leaf-команды. Команда offline: без
провайдеров, ASR, сети и платных вызовов; она не пишет БД и не запускает
миграции, не создаёт пустых sidecar в quiescent-доме и не удаляет живой
`-wal`/`-shm`.

```json
{
  "status": "success",
  "dry_run": false,
  "database": {"path": "...", "exists": true, "schema_version": 3},
  "attempts": 4,
  "totals": [
    {
      "currency": "RUB",
      "known_amount": "12.3400",
      "known_attempts": 2,
      "exact_attempts": 1,
      "non_exact_attempts": 1,
      "unknown_attempts": 2
    }
  ],
  "operations": [
    {
      "operation": "tts",
      "currency": "RUB",
      "known_amount": "12.3400",
      "known_attempts": 2,
      "exact_attempts": 1,
      "non_exact_attempts": 1,
      "unknown_attempts": 0
    }
  ],
  "completeness": "partial",
  "local_attempts_without_api_charge": 3
}
```

- Одна строка `attempts` — одна уникальная финансовая попытка: сумма считается
  ровно один раз, никогда как родитель + ребёнок. Повторный запрос статуса
  обновляет ту же строку (идемпотентно), поэтому не раздувает итог, а legacy
  run-total и per-chunk-цены взаимно исключают друг друга уже при импорте.
- Каждый `known_amount` — сумма сохранённых decimal-строк, накопленная
  `Decimal` (не binary float): `0.1 + 0.2` даёт `0.3`. `known_amount` включает
  non-exact legacy-lexeme и не утверждает их точность; `exact_attempts` и
  `non_exact_attempts` показывают разбивку, а `completeness` = `partial`, если
  есть cloud-unknown **или** non-exact попытка; иначе `complete`. `"0"` —
  реальный наблюдаемый ноль; `known_amount: null` в группе означает, что
  наблюдаемых сумм в ней нет (не выдуманный ноль).
- Группировка — по записанной валюте и операции. Отсутствующая валюта остаётся
  отдельным bucket `currency: null`, а не подставляется как RUB. `totals`
  агрегирует операции по валюте, `operations` — по паре (операция, валюта).
- `local_attempts_without_api_charge` считает попытки только из явного локального
  allowlist (`qwen-local`, `omnivoice-local`, `nemotron-local`, `faster-whisper`)
  без сохранённой суммы: локальное исполнение не даёт внешнего API-списания.
  Любой другой провайдер — потенциально платный, и облачный unknown остаётся в
  `unknown_attempts`.
- Чтение read-only: при отсутствии `-wal` это immutable-снимок без создания
  sidecar, поэтому quiescent-дом не теряет читаемость; при наличии `-wal`
  читаются его закоммиченные кадры, а основной файл/`-wal` не изменяются и
  sidecar не удаляются; SQLite может обновлять существующий `-shm`.
  Выбор reader'а — lock-free проверка на момент открытия:
  прогон, создавший `-wal` уже после неё, в этом снимке не виден, но значение
  никогда не выдумывается. Пустая БД → пустые `totals`/`operations`
  и `completeness: "complete"`; отсутствующая БД ничего не создаёт.
- malformed decimal в `cost` или небезопасная (URL/секрет-подобная) валюта
  падают fail-closed: exit `30` `HISTORY_COST_UNREADABLE`, значение не
  эхоится. Полный сценарий, prepared text, transcript, remote ID, пути и секреты
  не выводятся.

## `history resume ID` / `history sync ID` — DB-first продолжение

`history resume ID` и `history sync ID` восстанавливают **один уже
закоммиченный нативный TTS-прогон** (обычный не-диалоговый `polza-tts` /
`polza-chat-audio` / `openrouter-tts`, локальный Qwen (clone или instructed
preset/design) или `omnivoice-local` (preset bank или non-preset
auto/clone/design), записавший снимок в canonical SQLite) из его сохранённого
снимка и запускают тот же нативный исполнитель, что и `generate --resume`. `ID`
— внутренний `run_uuid`; метка не разрешается, потому что resume и sync меняют
один конкретный прогон. Оригинальный файл `script.md` **не перечитывается**: текст
берётся из committed `tts_script` источников, а сохранённый путь остаётся только
полем `script` в совместимом экспорте.

```bash
voiceover history resume ID --json      # потенциально платно
voiceover history sync ID --json        # без нового оплаченного submit
```

- **Потенциально платный `resume`.** Пропускает части с проверенным
  converted chunk, локально пересобирает часть из валидного raw receipt,
  доводит известный Polza Media task id только GET-запросами и делает новый
  платный POST **только** для действительно unattempted части (в обычном порядке
  оплаты, с pre-submit маркером и обычной проверкой identity). Для прогона с
  сохранёнными настройками локальных таймингов он дополнительно завершает
  недостающий локальный faster-whisper шаг (без TTS POST/GET), а проверка
  доступности модели происходит до любого платного POST. `--overwrite` не
  принимается вообще (exit `2`), чтобы не удалить принятые платные доказательства.
- **Бесплатный `sync`.** Никогда не начинает новый оплаченный submit и **никогда
  не запускает локальную модель таймингов**: недостающие локальные тайминги
  остаются незавершёнными (`timing.complete = false`) для явного resume. На
  завершённом прогоне починка ограничивается четырьмя совместимыми JSON из
  committed DB-вида (без API-ключа, провайдера и GET); при валидном raw
  receipt/`raw_saved` — локальная пересборка; при известном Media ID —
  GET-only добор с ленивым построением провайдера/ключа; синхронные маршруты без
  Media ID восстанавливаются только локально. Неотправленная (unattempted) часть
  или неподтверждённый submit блокируют (`NATIVE_SYNC_PAID_SUBMIT_REQUIRED` /
  `PAID_SUBMIT_UNCONFIRMED`, exit `30`) до чтения ключа и построения провайдера,
  без единого POST; незавершённая локальная часть `omnivoice-local` блокируется
  отдельным `NATIVE_SYNC_LOCAL_SYNTHESIS_REQUIRED` (exit `30`) без запуска
  локальной модели; остаток прогона молча не исполняется.
- **Один писатель и CAS.** Оба глагола идут через тот же межпроцессный run lock и
  CAS-резервацию, что и `generate`; занятый lock → exit `30` `NATIVE_RUN_LOCKED`.
- **Никаких новых прогонов.** Вызов привязан к запрошенному UUID, поэтому не
  создаёт новую строку прогона и не переключается на другой прогон того же
  `run_root` (`NATIVE_HISTORY_RUN_MISMATCH`). Неизвестный/отсутствующий UUID,
  отсутствующая БД и ненативный (импортированный/legacy/другой provider)
  снимок падают **до** любой записи: exit `2` `HISTORY_RUN_NOT_FOUND` /
  `HISTORY_INVALID_RUN_ID` или exit `30` `NATIVE_HISTORY_UNSUPPORTED`; дом, БД и
  строка прогона не создаются. Завершённый прогон, чей финальный аудиофайл
  пропал или изменён, падает fail-closed без пересборки (exit `50`
  `NATIVE_FINAL_AUDIO_MISSING`).
- **Приватность.** Успешный JSON — только метаданные (`run_uuid`, `revision`,
  `mode`, пути артефактов, длительность, cost, а для таймингового прогона
  `timing.complete`). Полный сценарий, prepared text, transcript, raw snapshot,
  подписанные URL и секреты не печатаются.

Эти два глагола реализуют `history resume ID` и `history sync ID`; live/listening
приёмка при этом не проводилась. `history costs` —
read-only offline-учёт расходов (см. выше). Локальные ASR/`verify-tts` и
облачные timing-маршруты, как и облачный dialogue-QA `xai-stt`, сохранены в
канонической истории (см. разделы выше). Оба нативных диалоговых маршрута
описаны ниже.

## Gemini Dialogue (machine-facing)

`format: dialogue` — канонический provider-independent двухголосый диалог.
`gemini-dialogue` сохраняется как compatibility alias с идентичным планом.
State/manifests всегда пишут `format: dialogue`. Режиссура и prompting живут в skill tree
(`docs/skills/voiceover-pipeline/`).

### Статус и граница доказательств

- OpenRouter `/api/v1/audio/speech` поддерживает **один голос на запрос**
  (одно top-level `voice`); поле `multi_speaker_voice_config` не
  документировано и никогда не отправляется.
- Исполнение `turn-by-turn-v1`: один request на реплику, с одним
  документированным `voice` и только `model`, `input`, `voice`,
  `response_format`. `input` byte-equals текущему turn text; `Alias:`, vibe,
  profiles, cast map и соседние turns отсутствуют.
- Для OmniVoice admission/runtime происходит один раз; turn вызывает bound
  bank profile. Два разных profile ID с одинаковым `reference_sha256`
  отклоняются до native call.
- Offline contract покрыт детерминированными tests. Это не доказывает
  слышимое различие: прошлый OpenRouter live attempt FAILED/BLOCKED, а новый
  OpenRouter и OmniVoice audible PASS остаются отдельными human gates.

### Frontmatter schema

```yaml
format: dialogue
language: ru
model: google/gemini-3.1-flash-tts-preview
speakers:
  <Alias1>:
    display_name: <строка>
    voice: <Gemini voice>
    profile: <строка>
  <Alias2>:
    display_name: <строка>
    voice: <Gemini voice>
    profile: <строка>
vibe: <строка>
allowed_tags:
  - <tag>
max_chunk_bytes: 3500
```

- `speakers` — ровно два алиаса; алиасы alphanumeric без пробелов.
- `voice` — из allowlist `GEMINI_TTS_VOICES`; два голоса обязаны быть
  различными.
- `allowed_tags` — опционально; по умолчанию полный safe-tag set.
- `max_chunk_bytes` — опционально; default `3500`, hard limit `4000`.
- `vibe` + speaker `profile` остаются validation/planning metadata с лимитом
  `4000` UTF-8 bytes (`STYLE_PROMPT_TOO_LARGE`), но OpenRouter synthesis input
  не меняют.

### Ограничения

- OpenRouter использует только `openrouter-tts` +
  `google/gemini-3.1-flash-tts-preview`; другая модель — `MODEL_NOT_GEMINI_TTS`.
- OmniVoice использует `omnivoice-local --mode preset --voice-bank` и profile
  IDs из admitted catalog. Обычный single-speaker OmniVoice сохраняет один
  native session; dialogue не смешивает его turns.
- Top-level `--voice` — производная совместимость: равен голосу первого
  спикера для OpenRouter. Явный конфликтующий `--voice` отклоняется (exit
  `2`) до создания провайдера.
- `--speaker-voice <Alias>=<Voice>` (repeat) переопределяет голос спикера;
  после overrides действуют те же правила: ровно два и различны; OmniVoice
  также требует разные fingerprints.
- **Недокументированный `multi_speaker_voice_config` в payload не
  отправляется и не поддерживается.** Один запрос — одна реплика (turn) —
  один top-level документированный `voice` (в рамках плана turn-by-turn).

### JSON и exit codes

- `validate --format dialogue --agent --json` — один JSON-объект
  отчёта (поля `status`, `valid`, `speaker_voice_map`, `chunk_reports`,
  `errors`, `warnings`), exit `0` даже при `valid: false`.
- Невалидный диалог при `generate --json` — один объект
  `{"status": "error", "error": "<первая ошибка>", "code": 2, "details": <отчёт>}`,
  exit `2`, до провайдера и платного запроса.
- `--json` + `--json-events` — несовместимы, отклоняются (exit `2`).
- OpenRouter dialogue без явного `--tts-quality-provider` отклоняется exit `2`
  до платного запроса.

### Resume identity

`run_state.json` хранит `synthesis_identity`: SHA-256 канонического UTF-8
JSON с strategy, provider/model, alias→voice/profile/fingerprint, hashes
упорядоченных turns и пауз, style/prompt hashes и trim policy; OmniVoice также
включает mode, seed, steps и guidance. Изменение каста, fingerprint, текста,
порядка, паузы, trim/model/prompt или OmniVoice parameters даёт exit `30`
до provider call. Старое dialogue state без `synthesis_identity`, равно как
orphan dialogue MP3 без trusted state, fail closed.

### Артефакты

- `chunks.json` и run manifest: `script_format: dialogue`, `speaker_voice_map`
  и один public receipt на turn: `turn_index`, `speech_duration_ms`,
  `audio_sha256`, `start_ms`, `end_ms`, `pause_after_ms`, speaker/voice и
  доступный fingerprint. Transcript, reference paths/text и voice-bank private
  data отсутствуют.
- `tts_quality.json` и поле `tts_quality` manifest хранят per-turn audio/text
  hashes, counts, strict similarity/repetition metrics и ASR identity без
  transcript/expected text. FAIL сохраняет receipt и даёт exit `60` до concat.
- `run_state.json` хранит trusted synthesis receipt без private dialogue text.
  Final concat использует только quality-PASS turn files строго в плане и после speech trim
  материализует локальную PCM тишину: 250/600/0 ms.

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

Локальный маршрут (`--asr-provider` или `--timing-provider faster-whisper`)
дополнительно сохраняет прогон операции `timings` в каноническую историю и
добавляет блок `history` (см. раздел «Локальные ASR / timings / verify в
канонической истории»). Облачные `--timing-provider` остаются legacy и блок
`history` не добавляют.

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
  ASR runtime. `auto` остаётся env-driven (`VOICEOVER_AUDIO_CPP_BINARY` и т.п.) и
  сам маршрут не переключает. Оба зарегистрированных provider-а (`qwen-local`,
  `nemotron-local`) чтут явный выбор детерминированно: `--runtime python` берёт
  их Python route, `--runtime audio-cpp` — native route, и при ошибке не
  переключаются на другой runtime. Explicit `audio-cpp` требует настроенного
  native audio.cpp package (и CUDA для Qwen) и иначе fail closed до factory;
  детали — в разделах провайдеров.
- Публичных флагов `--prompt`, `--glossary` или числового phrase
  boost нет. В API typed `ASRContextHints` различает `context_text`, glossary
  profile/digest/selected terms, `ASRPhraseHint` с силой `mild|normal|strong` и
  adapter-specific `initial_prompt`; они не записываются в стандартный receipt.
- `timestamp_mode: "none"` означает обычный text-only ответ adapter. `segments`
  или `words` появляются только при заявленных provider capabilities;
  timestamps имеют origin `native`, `forced` или `chunked`. `chunked` —
  маркер внешней long-form оркестрации, а не claim о native/forced word
  alignment; text-only long-form сегменты при этом не несут acoustic span.
  Text-only ответ не допускается к SRT.

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
    "model_path": null,
    "device": "cpu",
    "compute": "auto",
    "measurements": {}
  }
}
```

Если native ASR route запросил word timestamps, `execution` дополнительно
может содержать `raw_timestamp_entries`: неизменённые записи runtime до
нормализации в canonical VOP words. Поле отсутствует у text-only результатов.

Локальный `transcribe` также сохраняет наблюдаемый результат в каноническую
историю (операция `asr`) и при успешном сохранении добавляет в `--json` только
безопасный блок `"history": {"saved": true, "run_uuid": "..."}`; остальные
поля и exit code не меняются. Сбой сохранения, отключённая история и приватность
описаны в разделе «Локальные ASR / timings / verify в канонической истории».

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
`covered_duration_s`, `processed_duration_s`, `source_sha256`,
`coverage_verified` и manifest каждого фрагмента (`input_*`, измеренный
`output_duration_s`, его delta и допуск, `output_sha256`, `output_status`,
`coverage_*`, status и counts). Границы покрытия (`coverage_*`) остаются
техническим доказательством в `execution.long_form`, а не acoustic span:
text-only long-form сегменты не получают `start_s`/`end_s`, word-timed
сегменты несут наблюдаемые границы удержанных слов, а провайдерские
segment-времена сохраняются только когда их текст доказуемо отображается в
удержанный транскрипт. `source_sha256` — хэш исходного аудио, а
`output_sha256` — хэш извлечённого фрагмента; изменение исходника во время
выполнения fail-closed. Все фрагменты обязаны совпадать по provider, model,
runtime, `model_revision` и `model_path`; расхождение отклоняется до merge.
Локальный `transcribe` сохраняет в каноническом SQLite snapshot `observed_spans`:
`origin`, единицу `ms`, model/path/revision, UUID ровно сохранённого текстового
источника, SHA-256 его транскрипта и целочисленные
`char_start/char_end/start_ms/end_ms` только для наблюдаемых слов или
provider-сегментов с `native`/`forced` origin. При сохранении long-form history
хэш исходного аудио сравнивается с `source_sha256` его receipt: если файл
изменился после распознавания, завершённый run с противоречивыми SHA не пишется.
Неизвестный либо `chunked` span остаётся `null`, а внешний coverage не
превращается в речевую метку. Публичный `transcribe --json` по-прежнему отдаёт
совместимые `start_s/end_s`; это внутренний формат истории.
Для decoded MP3/codec seek-timebase границы допускается только bounded 0.10 s delta
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
provider. Его default model — `Qwen/Qwen3-ASR-0.6B`; тем же provider ID явно
выбирается `Qwen/Qwen3-ASR-1.7B`. Каждый размер владеет собственной локальной
директорией весов, поэтому запрошенные 1.7B не могут быть обслужены каталогом
0.6B, а неизвестный model ID отклоняется до загрузки. Runtime устанавливается
явно, без автоматической загрузки модели:

```powershell
voiceover list asr-providers --json
voiceover transcribe --audio recording.wav --provider qwen-local --model Qwen/Qwen3-ASR-1.7B --json
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
  model and Hugging Face cache directories, выбранными явно, и передаёт эти
  пути с `local_files_only=True` в runtime, поэтому он не может скачать модель
  или использовать root cache. Корень ассетов разрешается в порядке: переменные
  окружения `VOICEOVER_QWEN_ASR_MODELS_ROOT`, `VOICEOVER_QWEN_ASR_CACHE_DIR`,
  `VOICEOVER_QWEN_ASR_REVISION`; затем `settings.toml` `[asr.qwen_local]`
  `models_root`/`cache_dir`/`revision`; затем legacy
  `/media/v/storage/voiceover-pipeline/qwen-asr` с одним deprecation warning.
  В корне лежат `models/<Qwen3-ASR-0.6B|Qwen3-ASR-1.7B|Qwen3-ForcedAligner-0.6B>`
  и `huggingface-cache`. Каталог `models/<selected-name>` должен доказывать выбранную
  модель: alias на другой размер или на чужой Hugging Face snapshot отклоняется до
  загрузки. Явно заданная revision должна совпасть с каталогом Hugging Face
  snapshot, иначе admission падает до загрузки. `doctor` проверяет ту же local
  admission и сообщает availability, когда хотя бы одна из выбираемых моделей
  размещена, без загрузки модели и без оценки GPU.
- Явный `--runtime python` выбирает Python route, а явный `--runtime audio-cpp` —
  native route, независимо от выставленных `VOICEOVER_AUDIO_CPP_*` переменных;
  `auto` сохраняет прежний выбор по окружению. Перед long-form extraction explicit
  Python route проверяет admission именно выбранной модели, а explicit audio.cpp
  route — pinned inventory, поэтому неподдерживаемый размер отклоняется раньше, чем
  извлечён первый чанк.
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
  static contract не доказывает Windows inference/readiness. Эта pinned
  inventory обслуживает только `Qwen/Qwen3-ASR-0.6B`: другой выбранный размер
  отклоняется до любого runtime invocation с exit code `10` и предложением
  `--runtime python`.
- При отсутствии required local model/cache directories для выбранной модели
  selected Python route also returns exit code `10`, без сетевой попытки.
  Receipt и history сохраняют фактически выбранный model ID, разрешённый effective
  путь весов (`ASRExecutionReceipt.model_path`) и observed revision: если каталог
  является Hugging Face snapshot, его revision записывается даже без заданного pin.
  Если выбранный root/cache/revision меняется после загрузки модели, следующий
  runtime-вызов отклоняется, поэтому long-form чанки не могут обслуживаться разными
  весами. Загрузка и word-режим используют канонический путь весов, ForcedAligner
  и cache, зафиксированный на admission: retarget симлинка выбранного размера или
  aligner во время загрузки не подменяет загруженные веса, и такой retarget
  отклоняется до следующего runtime-вызова.

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

Для **legacy** маршрута timing failure при `--with-timings` — это hard error
(code 40), но MP3 сохранён:

```json
{
  "status": "error",
  "error": "Voiceover generated but timing extraction failed: ...",
  "code": 40
}
```

Для **нативного** обычного `polza-tts` / `polza-chat-audio` / `openrouter-tts` прогона с
`--timing-provider faster-whisper` сбой локальных таймингов — это exit `50` с
фиксированным `details.error_code` (`NATIVE_TIMING_FAILED` или
`NATIVE_TIMING_HISTORY_FAILED`); завершённое аудио, оплаченный raw и стоимость
сохраняются, а недостающие тайминги завершает явный `--resume` (см. «Нативная
история...»).

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

## Поиск по истории и offline-индекс (S08)

S08 добавляет производный полнотекстовый слой поверх канонической SQLite-истории.
Поиск работает по **сохранённым текстам, связанным с аудио** (сценарий, ASR
транскрипт, verify-транскрипт и короткая метка запуска), а не по звуковому
сигналу. Он не требует API-ключей, FFmpeg, Torch или `sqlite-vec`, не делает
сетевых и платных вызовов и не сканирует пользовательские файлы: индекс
пересобирается только из SQLite.

```bash
voiceover search "индексы SQLite" --mode lexical --limit 10 --json
voiceover search "транзакции" --mode lexical --kind asr_transcript --json
voiceover search "спокойный" --mode lexical --scope directions --json
voiceover index status --json
voiceover index build --json
voiceover index rebuild --json
```

### Поведение `search`

- Пользовательская строка обрабатывается как набор **буквальных** слов: кавычки,
  дефисы, пунктуация и SQL-подобные фрагменты разбираются в токены и не являются
  ни произвольным выражением MATCH, ни SQL-инъекцией. Пустой по токенам запрос →
  exit `2` `SEARCH_EMPTY_QUERY`.
- Роли и scope: `label`, `speech` (сценарий), `asr` (ASR и verify транскрипты),
  `directions` (режиссёрские инструкции). `--scope` — `speech` (default,
  speech+asr+label), `directions` (добавляет directions) или `all`. Приватный
  ASR-контекст (`asr_context`, подсказка-промпт) в индекс **не попадает**.
- Поисковая нормализация `ё → е` и регистра применяется **только** к производному
  индексу, не изменяя сохранённый текст: сниппет срезан из оригинала. Это не
  морфологический поиск (формы слова не гарантированно совпадают).
- `--kind`, `--role`, `--run`, `--operation`, `--provider`, `--since`, `--until`
  применяются в SQL **до** `--limit`; `--since`/`--until` задают строго
  нуль-дополненные даты `YYYY-MM-DD` включительно (включая весь день `--until`);
  `--limit` — `1..500` (default `20`). `--provider` выбирает прогоны с
  попыткой этого provider до `--limit`, но сам по себе не утверждает, что
  каждый найденный текст принадлежит именно этой попытке.
- Результат содержит `run_uuid`, `user_label`, `kind`, `role`, `text_source_uuid`,
  `part_uuid`, `artifact_uuid`, `chunk_index`, `char_start/char_end`,
  `start_ms/end_ms` (только если временной диапазон реально известен, иначе
  `null`), `text_hash`, `provider`, `model`, `created_at`, `snippet` и `audio` —
  ссылку на аудио с `availability` (`present`/`missing`). `provider`/`model`
  берутся только из подходящей типу текста и части **однозначной** попытки
  (сценарий TTS, transcript ASR, verification quality ASR); если таких попыток
  несколько с разными маршрутом/моделью или нет, поля равны `null`, а не
  произвольной попытке прогона.
- ASR/timings/verify-транскрипт и метка ссылаются на собственное
  `asr_source_audio`: для нового платного прогона ссылка — артефакт, записанный
  вместе с durable `submitting` marker до POST; для более старого платного прогона
  без такого артефакта — сохранённый абсолютный путь из его канонического
  identity-snapshot (`audio.artifact_uuid: null`). Завершённый TTS-прогон
  ссылается на итоговое аудио; в незавершённом — только на аудио **той же** части,
  а не соседней, либо `audio: null`, если своё аудио ещё не создано.
  Отсутствующий локальный файл, в том числе удалённый **после сохранения**, не
  скрывает текст из истории: результат
  возвращается с `audio.availability = "missing"`. Сохранённый как `missing`
  артефакт не повышается до `present` без подтверждения байтов. Временные метки
  не выдумываются: FTS берёт диапазон только для точного `asr_transcript` с
  совпавшими UUID источника и SHA-256 текста из канонического SQLite snapshot
  и лишь когда **все**
  речевые символы поискового чанка покрыты наблюдаемыми native/forced span-ами.
  Частичные и text-only чанки, другие роли, старые и импортированные записи,
  а также локальный `timings` без такой посимвольной привязки показывают `null`.
  `index rebuild` воспроизводит диапазоны из SQLite без модели, аудио и сети.
- Отсутствующая БД или непересобранный индекс (`search_chunks` ещё нет) → exit `0`
  с пустым результатом и предупреждением про `voiceover index build`; ничего не
  создаётся. Старые тексты после upgrade v2→v3, не охваченные немедленной
  индексацией новых источников, pending-источники, пропущенные label-чанки,
  неполные тексты без сохранённого содержимого и несовпадение версии chunker
  отмечаются отдельно в `warnings`; `index build` добирает индексируемые тексты,
  а `index rebuild` исправляет версию. Источник без сохранённого текста
  невозможно восстановить одной пересборкой.
- Режим берётся из явного `--mode`, а при его отсутствии — из несекретного
  `<CWD>/settings.toml`, секция `[search] default_mode` (`lexical`, `semantic`
  или `hybrid`; отсутствие файла/секции/ключа → `lexical`). Режим никогда не
  читается из `.env`: `search` не читает ключ или env-файл.
- Явный `--mode` побеждает всегда, даже когда консультируемый `settings.toml`
  испорчен или содержит недопустимое значение. Если `--mode` не задан и файл
  нечитаем, не является таблицей или `default_mode` не входит в три допустимых
  значения, команда fail-closed **до** открытия/создания базы и любого
  provider/model пути: exit `2`, `details.error_code = SEARCH_SETTINGS_INVALID`,
  одно фиксированное сообщение без пути и содержимого файла.
- Настроенный `semantic`/`hybrid` (`--mode` или `[search] default_mode`)
  принимается reader'ом настроек и отклоняется самим `search` с честным
  `SEARCH_MODE_DEFERRED` (exit `2`) **до** открытия/создания базы, без
  embeddings и provider: semantic/hybrid и оба embedding-backend относятся
  только к [плану следующего релиза](plans/2026-10-01-semantic-search-next-release-plan.md)
  (`S09 DEFERRED`), а доступный сейчас слой — только лексический FTS5.

### Поведение `index`

- `index status` открывает БД read-only (не создаёт новые sidecar; при чтении
  действительного WAL SQLite может обновить существующий `-shm`) и отдаёт
  честные счётчики: `sources_indexable`, `sources_indexed`, `sources_pending`,
  `sources_private_excluded` (приватный `asr_context`), `sources_incomplete`
  (только hash, без текста), `indexed_chunks`, `label_runs`, `labels_missing`,
  `chunker_version`, `built_chunker_version`, `complete`, `needs_rebuild`.
- `complete: false`, пока есть pending, неполный импортированный корпус
  (`sources_incomplete > 0`), пропущенная короткая метка прогона
  (`labels_missing > 0`) или индекс собран другой версией chunking. Старый
  импортированный неполный корпус **не** объявляется полностью проиндексированным.
- `index build` доиндексирует только отсутствующее (idempotent), `index rebuild`
  полностью пересобирает производный слой из `text_sources` и `runs`. Обе команды
  идут одной `BEGIN IMMEDIATE` транзакцией: сбой оставляет прежний индекс
  неизменным. Приватный `asr_context` не считается вновь проиндексированным
  при повторном build/rebuild. Удаление канонического прогона/текста убирает
  его FTS-термы через SQL-триггер, не оставляя скрытых производных копий после
  каскада.
- Обычное сохранение текста индексируется сразу. Если производная запись не
  удалась, канонический текст остаётся закоммиченным, источник помечается
  `search_index_pending` с предупреждением, и `index build` доберёт его. Для
  label-only прогона без текстового источника `index status` обнаружит
  пропущенную метку через `labels_missing` и `index build` восстановит её.
- Числа чанков: около 1200 Unicode-символов с перекрытием до 150 по границам
  абзацев/предложений; это настройки поискового слоя, они **не меняют** TTS-части,
  аудио и стоимость синтеза. Версия алгоритма (`chunker_version`) хранится вместе
  с индексом.

### `search --json`

```json
{
  "status": "success",
  "dry_run": false,
  "database": {"path": "...", "exists": true, "schema_version": 3},
  "query": "индексы SQLite",
  "mode": "lexical",
  "scope": "speech",
  "filters": {"kind": null, "role": null, "run_uuid": null, "operation": null, "provider": null, "since": null, "until": null},
  "limit": 20,
  "count": 1,
  "results": [
    {
      "run_uuid": "...",
      "user_label": "prod",
      "operation": "asr",
      "kind": "asr_transcript",
      "role": "asr",
      "text_source_uuid": "...",
      "part_uuid": null,
      "artifact_uuid": null,
      "chunk_index": 0,
      "char_start": 0,
      "char_end": 42,
      "start_ms": null,
      "end_ms": null,
      "text_hash": "...",
      "provider": "qwen-local",
      "model": null,
      "created_at": "...",
      "snippet": "...",
      "audio": {"artifact_uuid": "...", "role": "asr_source_audio", "path": "...", "path_kind": "external_absolute", "availability": "present"}
    }
  ],
  "warnings": []
}
```

### `index status --json`

```json
{
  "status": "success",
  "dry_run": false,
  "available": true,
  "database": {"path": "...", "exists": true, "schema_version": 3},
  "chunker_version": 1,
  "built_chunker_version": 1,
  "last_build_at": "...",
  "last_build_mode": "build",
  "sources_total": 3,
  "sources_indexable": 2,
  "sources_indexed": 2,
  "sources_pending": 0,
  "sources_private_excluded": 1,
  "sources_incomplete": 0,
  "label_runs": 1,
  "labels_missing": 0,
  "indexed_chunks": 3,
  "complete": true,
  "needs_rebuild": false
}
```

### Коды ошибок S08

Все новые причины передаются стабильной строкой `details.error_code` при обычных
числовых кодах `2`/`30`/`50`: `SEARCH_EMPTY_QUERY`,
`SEARCH_MODE_DEFERRED`, `SEARCH_SETTINGS_INVALID`,
`SEARCH_INVALID_LIMIT`, `SEARCH_INVALID_SCOPE`, `SEARCH_INVALID_ROLE`,
`SEARCH_INVALID_KIND`, `SEARCH_INVALID_RUN`, `SEARCH_INVALID_DATE` (все exit `2`),
плюс унаследованные `HISTORY_DATABASE_*`/`HISTORY_HOME_*` при чтении/записи БД.

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
| `--resume` + `--overwrite` вместе | Ошибка exit code 2 до любого удаления: флаги взаимоисключающие |
| Папка существует + `--overwrite` + подтверждённый `chunk_*.mp3` | Ошибка без `--confirm-delete-paid-audio` (exit code 30) |
| Папка существует + `--overwrite` + `pending_attempt` (в т.ч. рядом с подтверждённым `chunk_*.mp3`) | Ошибка exit code 30: проверка маркера идёт раньше подтверждения удаления, удаление запрещено даже с `--confirm-delete-paid-audio`, нужен другой `--run-id` |
| Папка существует + `--overwrite` + `run_state.json` не-объект или нечитаем | Ошибка exit code 30: состояние не доказывает отсутствие платного submit |
| Папка существует + `--resume` + присутствует `pending_attempt` или состояние нечитаемо | Ошибка exit code 30 `PAID_SUBMIT_UNCONFIRMED` до необязательных preflight'ов качества/таймингов, чтения ключа, создания провайдера, запроса цен и проверок идентичности (кроме валидного raw receipt или известной `polza-tts` `/media` задачи, см. выше) |
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
- OmniVoice local uses the Linux container route or the native-Windows factory; native Windows clone/design/bank paths are accepted (см. `docs/reports/2026-08-21-native-windows-omnivoice-voice-bank-acceptance.md`)
- OmniVoice local ASR explicit `--runtime audio-cpp` проходит валидацию, но fail closed (exit code `2`) как not implemented — планируется, не является рабочей capability
- Первый Whisper запуск скачивает модель (~486 MB) из HuggingFace
- При `--with-timings` ошибка Whisper — hard failure (code 40), но MP3 уже сохранён
- `polza-chat-audio` не делает автоматический второй платный POST: `--fallback-voice` принимается только для совместимости, а отказ первого голоса завершает запуск (exit code 30) без автоматической смены голоса
