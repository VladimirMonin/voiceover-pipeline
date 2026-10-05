---
name: voiceover-pipeline
description: >
  Используй ВСЕГДА для озвучки, голосовых отчётов, обзоров и рассказов голосом,
  TTS, аудио для видео, подкаста или Remotion, а также для таймингов,
  субтитров и распознавания речи через voiceover-pipeline CLI. Локальные
  провайдеры Qwen3-TTS и OmniVoice работают на GPU; явно названный
  пользователем провайдер не подменяется. Триггеры: озвучь, голосовой отчёт,
  обзор голосом, расскажи голосом, запиши аудио или новости, на нашей
  видеокарте, voiceover, TTS, тайминги, whisper timing, аудио для видео,
  подкаст, generate audio, timings for Remotion, voiceover-pipeline, выбери
  провайдера, сравни модели TTS, голос для озвучки, format: voiceover,
  dialogue, --resume, status run, concat partial audio, Gemini prompting,
  audio tags.
---
# Voiceover Pipeline — навык агента

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ. Сначала [безопасность](docs/03-security-and-secrets.md):
> реальный `.env` не трогать, сеть/платные вызовы — только с разрешения владельца.
> Детали в docs/ — по необходимости; запись файлов только через инструменты редактирования.
> **Совместимость:** пакет 0.8.0, skill revision 2026-10-05 (release).
> **Версионный лог:** [docs/00-version-log.md](docs/00-version-log.md)

## Справка из пакета (S10)

`voiceover help [TOPIC] [--raw | --json]` читает один атомарный Markdown из
установленного пакета: без ключей, `.env`, рабочего каталога, FFmpeg, GPU, истории
и сети. Без темы печатается `index` со списком тем. Имена тем — строгий lowercase
dotted identifier; неизвестная тема — exit `2`. Актуальные провайдеры, модели и
флаги берите из `voiceover list ...`, `voiceover <cmd> --help` и этих тем.

| Тема | О чём |
|---|---|
| `start.quick`, `runs.resume` | Ключи, данные, первые команды, resume/status/sync |
| `speech.simple`, `speech.parts`, `speech.legacy` | Одна реплика, явные части, прежние сценарии |
| `asr.transcribe`, `history.find`, `history.costs` | Распознавание, история, расходы |
| `search.lexical`, `search.semantic`, `cli.json`, `providers.polza` | FTS5 и отложенный semantic/hybrid, JSON-контракт, маршруты Polza |

## Назначение

Научить агента самостоятельно устанавливать voiceover-pipeline и его
пререквизиты (Python, UV, FFmpeg), проверять окружение, создавать болванки
проекта, генерировать озвучку + Whisper-тайминги из Markdown-сценариев
и отдавать готовые артефакты (MP3, timings.json, SRT, manifest.json)
для Remotion, монтажа или подкастов.

Без этого навыка агент может пытаться вызывать TTS API вручную, оценивать
длительности по словам или просить пользователя выполнять terminal-команды.

## Режимы

| Режим | Когда | Порядок |
|---|---|---|
| **A: Bootstrap** | Проекта нет, CLI не установлен, нет ключей | Установка → .env.example → .gitignore → script.md → out/ |
| **B: Generate** | Сценарий готов, ключи есть | doctor → validate → generate → manifest.json |
| **C: Timings only** | Готовый MP3/Opus, нужны SRT/тайминги | timings --audio --timing-provider → .timings.json + .srt |
| **D: Troubleshoot** | Что-то сломалось | doctor --json → exit code → recovery |
| **E: Local hybrid** | Нужны локальные ASR/TTS через `audio.cpp` | inventory → doctor → explicit runtime → receipt → cleanup |
| **F: Two-speaker podcast** | «подкаст», «диалог», «два ведущих», «вопрос-ответ» | author script → validate → doctor → approval → generate → artifacts |
| **G: Short line / speech-parts** | «одна реплика», явные части с одним голосом на часть | text/speech-parts → preflight → approval → generate (native) |

## Когда навык должен срабатывать

**Должен:**
- «озвучь этот markdown-сценарий»
- «сделай voiceover для Remotion»
- «сгенерируй аудио и тайминги из скрипта»
- «нужно получить SRT из MP3»
- «поставь voiceover-pipeline и проверь что работает»
- «сделай подкаст из сценария»
- «сделай голосовой отчёт / обзор голосом / расскажи это голосом»
- «запиши рассказ или новости на нашей видеокарте»
- «сделай подкаст с двумя ведущими» (→ режим F, gemini-dialogue)
- «озвучь диалог мужчины и женщины» (→ режим F, gemini-dialogue)
- «сделай Q&A / вопрос-ответ двух спикеров» (→ режим F, gemini-dialogue)
- «voiceover generate с таймингами»
- «какие есть провайдеры/модели/голоса для TTS»
- «какие есть провайдеры для распознавания речи»
- «распознай аудио через облачный whisper»
- «транскрибируй подкаст через OpenRouter»
- «выбери дешёвую озвучку»
- «сравни качество TTS моделей»

**Не должен:**
- «объясни как работает git tag»
- «напиши сценарий для ролика, но не озвучивай» (только сценарий без артефакта — творческая задача)
- «сделай Mermaid-диаграмму»
- «отрендери Remotion-видео целиком»
- «установи Python» (если нет привязки к voiceover)

## Каталог файлов

| Приоритет | Файл | Читать когда |
|---|---|---|
| ВСЕГДА | [docs/00-version-log.md](docs/00-version-log.md) | Нужно знать версию CLI и историю изменений (не источник цен и доступности) |
| ВСЕГДА | [docs/03-security-and-secrets.md](docs/03-security-and-secrets.md) | До любого действия с .env или ключами |
| ВСЕГДА | [docs/02-install.md](docs/02-install.md) | Нужно установить CLI, зависимости или понять какую сборку выбрать |
| ВСЕГДА | [docs/01-concept.md](docs/01-concept.md) | Нужно понять что это и зачем |
| По ситуации | [docs/04-input-format.md](docs/04-input-format.md) | Нужно создать или проверить сценарий |
| По ситуации | [docs/05-providers-and-models.md](docs/05-providers-and-models.md) | Нужно выбрать TTS-провайдера, модель или голос |
| По ситуации | [docs/13-speech-recognition-providers.md](docs/13-speech-recognition-providers.md) | Нужно выбрать локальное/облачное распознавание и вид таймкодов |
| По ситуации | [docs/14-local-audio-cpp-models.md](docs/14-local-audio-cpp-models.md) | Нужны Qwen3-ASR, Nemotron, Qwen3-TTS, OmniVoice, benchmark или Windows boundary |
| По ситуации | [docs/06-commands-and-flags.md](docs/06-commands-and-flags.md) | Навигация по командам, ключи, exit codes, безопасность платных прогонов |
| По ситуации | [docs/07-artifacts.md](docs/07-artifacts.md) | Нужно понять что на выходе |
| По ситуации | [docs/08-workflows.md](docs/08-workflows.md) | Нужен готовый end-to-end сценарий |
| По ситуации | [docs/09-troubleshooting.md](docs/09-troubleshooting.md) | Что-то пошло не так |
| По ситуации | [docs/10-evaluation.md](docs/10-evaluation.md) | Проверить качество навыка |
| По ситуации | [docs/11-gemini-prompting.md](docs/11-gemini-prompting.md) | Нужна режиссура Gemini TTS, audio tags, эмоции, chunk limits |
| По ситуации | [docs/12-gemini-prompting-templates.md](docs/12-gemini-prompting-templates.md) | Нужны project-native Gemini examples, prompt templates, QA checklist |
| Примеры | [examples/](examples/) | Нужен образец сценария, .env.example, Remotion-поток или двухголосый подкаст |

## Обязательный быстрый алгоритм

1. **Безопасность прежде всего.** Прочитай `docs/03-security-and-secrets.md`.
   Создай только `.env.example` с placeholder-ами и убедись в `.gitignore`.
   Агент НЕ создаёт, не копирует, не читает и не печатает реальный `.env`;
   пользователь помещает ключи в приватный env-файл вне инструментов агента или
   использует уже существующие переменные окружения процесса. Порядок разрешения:
   непустое окружение процесса → явный `--env-file PATH` → необязательный
   path-only `VOICEOVER_POLZA_ENV_FILE`/`VOICEOVER_OPENROUTER_ENV_FILE` (заменяет
   CWD-файл для своего провайдера) → `<CWD>/.env`; поиска по родительским каталогам нет.
2. **Bootstrap проекта.** Создай болванки: `script.md` (если нет), `out/`,
   `.env.example` уже создан. Если CLI не установлен — поставь Python/UV/FFmpeg
   (если среда позволяет), затем выбери сборку по `docs/02-install.md`. Сетевые
   установки и загрузку моделей выполняй только с явного разрешения владельца.
3. **Выбор провайдера.** Если пользователь не указал — прочитай
   `docs/05-providers-and-models.md`, предложи варианты. По умолчанию:
   `polza-chat-audio` с `openai/gpt-audio-mini` или
   `polza-tts` с `openai/gpt-4o-mini-tts` (классический TTS с выбором голоса).
   Локальные `audio.cpp` routes не выбирать неявно: сначала прочитать
   `docs/14-local-audio-cpp-models.md`, проверить модель, platform route и GPU.
4. **Проверка окружения.** `voiceover doctor --provider <X> --with-timings [--timing-provider <Y>] --json`.
   Убедись что `workflow_ok: true`; запускай `doctor` только в одобренном
   окружении. `doctor` читает наличие ключа (печатает путь, не значение); при
   непригодном явном `--env-file` он даёт exit `0` и `status: success`, но
   `workflow_ok: false`.
   Если нужны таймкоды через облако: `--timing-provider groq-whisper` или `--timing-provider xai-stt`.
5. **Валидация сценария.** `voiceover validate --script "script.md" --json`.
   Если есть issues — покажи пользователю, не запускай генерацию.
   Для локального TTS цифры в произносимом тексте заранее преобразуй в слова;
   ID, пути, хэши и машинные десятичные дроби не отправляй модели как речь.
6. **Генерация аудио.** `voiceover generate --provider <X> --model <Y> --script "script.md" --run-id "prod" --json --resume`.
   Не используй `--overwrite` для платной генерации; если run оборвался — продолжай через `--resume`.
   Для длинного/выпускного TTS с доступным локальным ASR затем запусти
   `voiceover verify-tts --audio <mp3> --expected-file <script> --provider <ASR> --receipt <json> --json`.
   Exit `60` — quality FAIL; exit `0` всё равно требует человеческого прослушивания.
7. **Тайминги.** Предпочитай `generate --with-timings` в том же безопасном прогоне.
   Для обычного не-диалогового `polza-tts` / `openrouter-tts` с
   `--timing-provider faster-whisper` аудио и локальные субтитры пишутся одной
   командой в canonical history; локальная модель должна быть уже установлена и
   закеширована (неявного скачивания нет — иначе команда падает до платного POST),
   а сбой только таймингов даёт exit `50` с сохранённым MP3 и завершается явным
   `--resume`. `generate --with-timings --timing-provider groq-whisper|xai-stt`
   идёт через тот же платный boundary (маркер до POST, raw до парсинга, unknown
   цена); потерянный ответ не повторяется автоматически, а `history resume`
   воспроизводит уже сохранённое тело без второго POST. Если тайминги нужны
   отдельно — используй ДРУГОЙ `--output-dir`/`--run-id`,
   не перезаписывай папку платного прогона:
   `voiceover timings --audio "out/prod/<full>.mp3" --timing-provider <X> --output-dir "out" --run-id "prod-timings" --json`.
8. **Локальная история.** Локальные `transcribe`, `timings` (без облачного
   `--timing-provider`) и `verify-tts` сохраняют наблюдаемый результат в приватную
   SQLite-историю по умолчанию; `history list --operation asr|timings|verify`,
   `history show <uuid>`, `history costs` читают только метаданные (текст остаётся
   приватным, `has_content: true`). Отключить запись: `settings.toml` рядом с CWD
   с `[history] enabled = false` (облачный `timings` тогда fail-closed до
   запроса). Если сохранение не удалось, `--json` даёт `status: "partial"` и
   `history.error_code = "HISTORY_PERSISTENCE_FAILED"` с exit `50`. `history
   resume` может сделать новый платный POST. `history sync` не делает новый POST:
   он может восстановить локальный export/raw без сети или выполнить GET по
   известному Media ID; поэтому весь `history` нельзя считать офлайновым.
9. **Статус/артефакты.** `voiceover status --run-id "prod" --json`; прочитай `manifest.json`, `run_state.json`, `generation.log`.
   В receipt проверь `execution_source`: source kind, revision/dirty и package-tree SHA-256.

## Security-first правила

- **НИКОГДА не читай `.env`.** Даже чтобы проверить наличие ключа.
- **НИКОГДА не проси пользователя прислать ключ в чат.**
- Создай только `.env.example` с placeholder-ами `pza_...` и `sk-or-v1-...`. Реальный `.env` пользователь создаёт сам вне инструментов агента; не копируй шаблон в `.env` через shell и не читай его.
- Проверяй наличие ключей через `voiceover doctor --json`, а не через чтение файла.
- Убедись что `.gitignore` содержит `.env`.
- Дальше работай молча — не спрашивай ключи повторно.

## Структурные правила

- `SKILL.md` ≤300 строк — точка входа, не полный учебник.
- Каждый `docs/*.md` ≤300 строк — ровно одна тема на файл.
- Каждый `examples/*.md` ≤300 строк — образец, а не скрытая процедура.
- Запись файлов — через инструменты редактирования, не через shell.

## Граница навыка

| Навык ДЕЛАЕТ | Навык НЕ ДЕЛАЕТ |
|---|---|
| Устанавливает voiceover-pipeline + пререквизиты (Python/UV/FFmpeg), если среда позволяет | Устанавливает CUDA-драйверы, чинит системный PATH, делает низкоуровневый ремонт ОС |
| Создаёт .env.example (но не .env), .gitignore, script.md, out/ — болванки проекта | Не создаёт, не копирует и не читает .env или значения ключей |
| Проверяет окружение через doctor | Конфигурирует системный PATH |
| Валидирует Markdown-сценарий | Выдумывает несвязанный творческий контент |
| Генерирует озвучку через подтверждённый маршрут с paid-attempt marker, безопасным resume и manifest/log | Рендерит Remotion-видео |
| Извлекает тайминги через локальный faster-whisper ИЛИ облачные OpenRouter/Groq/xAI Whisper | Правит исходники voiceover-pipeline |
| Читает сохранённые ASR/timing/verify-метаданные через `history list/show/costs` | Печатает приватный transcript или ожидаемый текст из БД |
| Ищет по сохранённым текстам офлайн: `search` и `index status/build/rebuild` | Не делает семантический/векторный поиск (S09 отложен в план следующего релиза) и не сканирует файлы как транскрипт |
| Читает manifest.json → артефакты | Использует words-per-second при наличии timings |
| Объясняет провайдеров, модели и голоса (актуальные — в `voiceover list ...`) | Гарантирует будущие цены и доступность провайдеров |
| Диагностирует ошибки по exit codes | Правит исходники voiceover-pipeline |

## Режим F: Two-speaker podcast (dialogue)

На «подкаст», «диалог», «два ведущих», «вопрос-ответ» — канонический
`format: dialogue` (`gemini-dialogue` — совместимый alias; manifest и state
всегда пишут `dialogue`). OpenRouter: `openrouter-tts` +
`google/gemini-3.1-flash-tts-preview`; локальный путь — явный `omnivoice-local`
и admitted voice bank.

- OpenRouter принимает один top-level `voice` на запрос: генератор делает
  **один запрос на реплику (turn)**, голос по alias, `Alias:` вырезан,
  детерминированные паузы. `input` byte-equals turn text
  (style/profile/vibe/соседний контекст не отправляются).
- Ровно два различных спикера; перед concat нужен явный `--tts-quality-provider`,
  а выбор провайдера подтверждается пользователем до платного вызова.
- OmniVoice admission/runtime — один раз, но каждая реплика вызывает свой bound
  profile; одноголосый OmniVoice остаётся одним native session на прогон.
- Offline/mocked contract проверен, но audible acceptance OpenRouter и OmniVoice —
  отдельный live/listening human gate: не заявляй слышимое различие голосов или
  выпуск без двух PASS.
- Агент может написать диалоговый скрипт из темы и состава ведущих — это сценарий
  для запрошенного артефакта, а не выдуманный контент.
- Workflow `docs/08-workflows.md` → «Agent Podcast Workflow»; формат
  `docs/04-input-format.md`; пример `examples/gemini-dialogue-podcast.md`.

## Режим G: Короткая реплика и `speech-parts` (S06)

```bash
voiceover generate --provider polza-chat-audio --text "Добрый вечер." --voice ash --run-id greeting-01 --json
voiceover validate --script ./podcast.yaml --format speech-parts --json
voiceover generate --script ./podcast.yaml --format speech-parts --run-id podcast-01 --json
```

- Примеры `generate` платные: только после разрешения владельца и доказанного
  потолка расходов; зарегистрированный голос не подтверждает live-доступность.
- Непустой `--vibe` допустим только на маршруте, который несёт направление
  отдельным полем (Gemini 3.8 Flash/Flash-Lite на `polza-tts`/`openrouter-tts`);
  любой другой маршрут возвращает `BLOCKED_PROVIDER_CONTRACT` до ключа и POST.
- **Gemini 3.8 Flash / Flash-Lite:** `google/gemini-3.8-flash-tts` и
  `google/gemini-3.8-flash-lite-tts` — обычные admitted-модели `polza-tts` и
  `openrouter-tts` без opt-in: один scalar `voice` POST на часть и отдельное
  недокументированное `instructions`; устаревший
  `--allow-experimental-gemini-speech-parts` принимается, но ничего не требует
  ([подробности](docs/05-providers-and-models.md)). Ни multi-speaker POST,
  ни внешний счёт не подтверждены.
- `speech-parts` — строгий YAML (`version: 1`, `format: speech-parts`, непустой
  `parts` с `voice`/`text`); `provider`/`model` задаёт CLI, CLI `--voice`/`--vibe`
  с этим сценарием отклоняются без скрытых override; vibe — не произносимый текст.
- Прогон и resume идут через DB-first историю без повторного чтения YAML;
  все части проверяются до POST, поздняя слишком длинная часть не отправляется.
- `--audio-format {mp3,wav}` меняет только итоговый файл (default mp3); `wav` — только нативный маршрут.

## Поиск по сохранённому тексту (S08)

Когда нужно найти ранее сохранённую озвучку или расшифровку по словам:

```bash
voiceover search "индексы SQLite" --mode lexical --limit 10 --json
voiceover search "транзакции" --mode lexical --kind asr_transcript --json
voiceover index status --json
voiceover index build --json
```

- Поиск идёт по сохранённым текстам (сценарий, ASR/verify-транскрипт, метка),
  офлайн без ключей, FFmpeg, Torch и сети.
- Запрос — буквальные слова: кавычки, дефисы и пунктуация безопасны. По умолчанию
  ищется речь и распознавание, `--scope directions` добавляет инструкции; приватный
  ASR-промпт в индекс не попадает.
- Фильтры `--since`/`--until` включают названный день. Ссылка ASR/timings ведёт к
  исходному аудио; если файл пропал, текст остаётся в поиске, а
  `audio.availability` становится `missing`.
- Обычное сохранение индексируется сразу; старые тексты после upgrade и пропущенные
  метки помечаются в `warnings`/`index status`, `index build` доберёт их, а
  `index rebuild` пересобирает индекс только из SQLite.
- `--mode semantic|hybrid` — этап S09: команда честно отказывает (exit `2`), а не
  возвращает пустой результат. Режим по умолчанию берётся из несекретного
  `<CWD>/settings.toml` `[search] default_mode` (default `lexical`); настроенный
  `semantic`/`hybrid` отказывает до открытия базы, невалидный файл —
  `SEARCH_SETTINGS_INVALID`, явный `--mode lexical` побеждает. S09 вынесен в
  [план следующего релиза](../../plans/2026-10-01-semantic-search-next-release-plan.md).

## Чеклист готового навыка

- [ ] `SKILL.md` ≤300 строк, docs/ ≤300, examples/ ≤300
- [ ] Каталог ведёт в реальные файлы
- [ ] `description` покрывает реальные фразы пользователя (русский + English keywords)
- [ ] Есть trigger checks: should trigger / should not trigger / boundary
- [ ] Есть smoke tests (≥8 кейсов с assertions) и regression set
- [ ] Security-first правила на первом месте
- [ ] Все команды — bare (`voiceover ...`), кроме секции разработки
- [ ] `voiceover list` даёт зарегистрированные модели/голоса, не подтверждает тариф или верхнюю границу цены
- [ ] Локальный TTS-сценарий не содержит необработанных цифр и machine-readable ID
- [ ] Скорость/качество локальных моделей привязаны к конкретному receipt/corpus, а static и live evidence не смешаны
- [ ] Навык не привязан к одному агенту или IDE
- [ ] `docs/00-version-log.md` содержит совместимость с версией CLI
