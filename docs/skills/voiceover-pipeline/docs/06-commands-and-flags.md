# Команды: навигация, ключи и безопасность прогонов

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Канонический пользовательский справочник — упакованный `voiceover help` и
> `voiceover <cmd> --help`; исчерпывающие таблицы флагов и моделей здесь не дублируются.
> Артефакты — [07-artifacts.md](07-artifacts.md); провайдеры — [05-providers-and-models.md](05-providers-and-models.md).

## Канонические источники (не дублировать здесь)

- `voiceover help` — индекс тем, `voiceover help <topic> [--raw | --json]` — одна атомарная
  тема: `start.quick`, `speech.simple`, `speech.parts`, `speech.legacy`, `runs.resume`,
  `asr.transcribe`, `history.find`, `history.costs`, `search.lexical`, `search.semantic`,
  `cli.json`, `providers.polza`. Help не читает ключи/`.env` и не требует CWD, FFmpeg, GPU,
  истории или сети.
- `voiceover <cmd> --help` — точные флаги и defaults: `generate`, `timings`, `validate`,
  `history`, `search`, `index`, `doctor`, `list`, `status`, `concat`.
- `voiceover list providers --json`, `voiceover list voices --provider <id> --json`,
  `voiceover list timing-providers --json`, `voiceover list asr-providers --json` —
  зарегистрированные возможности, но не тариф или доступность внешнего API.
- `docs/agent-cli-contract.md` — машинный контракт: JSON stdout/stderr, события, коды, фильтры
  и денежный контракт истории, идентичность прогонов.

## Команды (обзор)

| Команда | Назначение |
|---|---|
| `doctor` | Проверить окружение (Python, FFmpeg, ключи, Whisper/CUDA) |
| `validate` | Проверить `markdown`/`voiceover`/`dialogue`/`speech-parts` сценарий без POST |
| `list` | Провайдеры, голоса, timing-модели и ASR-провайдеры |
| `split` | Разбить сценарий на чанки без генерации |
| `generate` | TTS + MP3 + опционально тайминги и quality-сверка |
| `timings` | Whisper-тайминги из готового аудио |
| `verify-tts` | ASR-сверка без публикации transcript |
| `status` / `concat` | Partial/resumable прогон; склейка chunks |
| `history` | `list`/`show`/`resume`/`sync`/`import`/`costs` (локальная SQLite) |
| `search` / `index` | Офлайн FTS5-поиск и индекс сохранённых текстов |
| `help` | Упакованные атомарные темы справки |

Все команды поддерживают `--json` (ровно один объект в stdout); события прогресса —
`generate --json-events` (NDJSON, взаимоисключимо с `--json`). Машинный формат —
`voiceover help cli.json`.

## Глобальные флаги и ключи

| Флаг | Назначение |
|---|---|
| `--env-file PATH` | Явный env-файл для ключей; указывается ДО подкоманды |
| `VOICEOVER_POLZA_ENV_FILE` / `VOICEOVER_OPENROUTER_ENV_FILE` | Необязательный path-only источник ключа Polza/OpenRouter (env-переменная процесса, не CLI-флаг) |

- Порядок разрешения ключа: непустая переменная окружения процесса → явный `--env-file`
  (валидный обычный файл, проверяется при чтении ключа) → необязательный path-only
  источник провайдера (`VOICEOVER_POLZA_ENV_FILE`/`VOICEOVER_OPENROUTER_ENV_FILE`,
  заменяет CWD-файл для своего секрета) → `<call-time CWD>/.env`. Родительские
  каталоги не ищутся; явный файл **заменяет** CWD-файл, а не сливается с ним.
- При процессном ключе содержимое файла не читается; непригодный явный путь (нет файла /
  не обычный / нет метаинформации) всё равно fail-closed при чтении ключа: exit `20`,
  одно redacted сообщение без пути.
- `help` не читает ключ/`.env` и не валидирует явный путь. `doctor` показывает наличие ключа
  и путь env-файла (не значение); при непригодном явном `--env-file` — exit `0`,
  `status: success`, но `workflow_ok: false`.
- Агент никогда не создаёт, не копирует и не читает реальный `.env`; им владеет пользователь
  (`docs/03-security-and-secrets.md`). Полный порядок — `voiceover help start.quick`.

## Exit codes (кратко)

| Код | Значение |
|---:|---|
| `0` | success |
| `2` | invalid args: файл, run-id, тема help, режим search, несовместимые флаги |
| `10` | missing dependency: локальный runtime или faster-whisper |
| `11` | нет `ffmpeg`/`ffprobe` |
| `20` | no key либо непригодный явный `--env-file` |
| `30` | provider/run error: ошибка API, существующий каталог без `--overwrite` |
| `40` | whisper/timing error (аудио сохранено) |
| `50` | output error или сбой сохранения полученного результата в историю |
| `60` | TTS quality failed (аудио сохранено) |

`details.error_code` несёт стабильный машинный код. Полная таблица и JSON-контракт —
`voiceover help cli.json` и [Agent CLI Contract](../../../agent-cli-contract.md).

## Безопасность платных прогонов и восстановление

- На часть отправляется не более одного платного POST; ошибка после отправки завершает запуск
  exit `30` без автоматического повтора или fallback. Перед submit пишется маркер
  `pending_attempt` (стадии `submitting`/`outcome_unknown`/`failed`/`raw_saved`, bounded receipt);
  пока маркер есть, `--overwrite` закрыт. Неопределённый платный submit не повторяется
  автоматически, а `--resume` блокируется (`PAID_SUBMIT_UNCONFIRMED`).
- Принятое платное аудио сохраняется в `raw/` до FFmpeg. Совпавший receipt позволяет `--resume`
  пересобрать часть локально **без POST/GET**; известный безопасный Media ID (`polza-tts`
  ElevenLabs) доводится GET-only. Нет ни raw, ни допустимого Media ID → `--resume` закрыт;
  новая попытка — только явным решением владельца и с другим `--run-id`.
- `--resume` и `--overwrite` взаимоисключающие: exit `2` до удаления и до сети.
- Нативный прогон (`polza-tts`, `polza-chat-audio`, `openrouter-tts` и два dialogue-маршрута)
  хранится в canonical SQLite; `run_state.json`/`chunks.json`/manifest — совместимый экспорт.
  Возврат к legacy JSON-writer запрещён, существующие legacy-каталоги не захватываются.
- Владение привязано к output-root двусторонним guard под общим run lock: `generate` не трогает
  paid-owned root, `timings` не трогает native-owned root (`PAID_TIMING_OUTPUT_OWNED`).
  Нечитаемая/несовместимая БД (либо отсутствующая БД при `history resume`),
  дескриптор без своего прогона, изменённая identity
  (`NATIVE_RESUME_IDENTITY_CHANGED`), занятый lock (`NATIVE_RUN_LOCKED`) и сбой экспорта
  (`NATIVE_EXPORT_FAILED`) — fail-closed без фоллбэка на JSON. Свежий прогон
  вправе создать новую приватную SQLite-историю.
- `history resume UUID` потенциально платный (может отправить unattempted часть).
  `history sync UUID` нового POST не делает: чинит export, локально пересобирает raw или
  доводит известный Media ID GET-ом, иначе fail-closed. Оба держат тот же lock, не создают
  БД/строку для неизвестного или legacy UUID и не принимают `--overwrite`.
- Локальные `qwen-local`/`omnivoice-local` используют cost-free `local_tts_chunk` (cost/status
  `null`) и неявно модель не скачивают. Облачные `timings`/quality-маршруты — платные.
- Маршруты и флаги — `voiceover help runs.resume`, `voiceover help asr.transcribe`,
  [13-speech-recognition-providers.md](13-speech-recognition-providers.md),
  [14-local-audio-cpp-models.md](14-local-audio-cpp-models.md).

## Правила `--run-id` и `--output-dir`

- `--run-id`: только `[a-zA-Z0-9._-]`; без `.`/`..`, разделителей пути, ведущих/замыкающих
  пробелов и точек, Windows reserved names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`,
  `LPT1`-`LPT9`) и `<>:"|?*`.
- `--output-dir`: не drive root, не home, не CWD; относительные или абсолютные пути вне них.
- Флаги команд и defaults берите из `voiceover <cmd> --help`, а не из этой страницы.
