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

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Детали в docs/ — одна тема на файл, читай по необходимости.
> Запись файлов ТОЛЬКО через инструменты редактирования, не через shell.
| **Совместимость:** voiceover-pipeline 0.6.0, skill revision 2026-08-21.
> **Версионный лог:** [docs/00-version-log.md](docs/00-version-log.md)

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
| **A: Bootstrap** | Проекта нет, CLI не установлен, нет .env | Установка → .env.example → .gitignore → script.md → out/ |
| **B: Generate** | Сценарий готов, ключи есть | doctor → validate → generate → manifest.json |
| **C: Timings only** | Готовый MP3/Opus, нужны SRT/тайминги | timings --audio --timing-provider → .timings.json + .srt |
| **D: Troubleshoot** | Что-то сломалось | doctor --json → exit code → recovery |
| **E: Local hybrid** | Нужны локальные ASR/TTS через `audio.cpp` | inventory → doctor → explicit runtime → receipt → cleanup |
| **F: Two-speaker podcast** | «подкаст», «диалог», «два ведущих», «вопрос-ответ» | author script → validate → doctor → approval → generate → artifacts |

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
| ВСЕГДА | [docs/00-version-log.md](docs/00-version-log.md) | Нужно знать версию CLI, актуальные цены, историю изменений |
| ВСЕГДА | [docs/03-security-and-secrets.md](docs/03-security-and-secrets.md) | До любого действия с .env или ключами |
| ВСЕГДА | [docs/02-install.md](docs/02-install.md) | Нужно установить CLI, зависимости или понять какую сборку выбрать |
| ВСЕГДА | [docs/01-concept.md](docs/01-concept.md) | Нужно понять что это и зачем |
| По ситуации | [docs/04-input-format.md](docs/04-input-format.md) | Нужно создать или проверить сценарий |
| По ситуации | [docs/05-providers-and-models.md](docs/05-providers-and-models.md) | Нужно выбрать TTS-провайдера, модель или голос |
| По ситуации | [docs/13-speech-recognition-providers.md](docs/13-speech-recognition-providers.md) | Нужно выбрать локальное/облачное распознавание и вид таймкодов |
| По ситуации | [docs/14-local-audio-cpp-models.md](docs/14-local-audio-cpp-models.md) | Нужны Qwen3-ASR, Nemotron, Qwen3-TTS, OmniVoice, benchmark или Windows boundary |
| По ситуации | [docs/06-commands-and-flags.md](docs/06-commands-and-flags.md) | Нужен полный CLI-справочник |
| По ситуации | [docs/07-artifacts.md](docs/07-artifacts.md) | Нужно понять что на выходе |
| По ситуации | [docs/08-workflows.md](docs/08-workflows.md) | Нужен готовый end-to-end сценарий |
| По ситуации | [docs/09-troubleshooting.md](docs/09-troubleshooting.md) | Что-то пошло не так |
| По ситуации | [docs/10-evaluation.md](docs/10-evaluation.md) | Проверить качество навыка |
| По ситуации | [docs/11-gemini-prompting.md](docs/11-gemini-prompting.md) | Нужна режиссура Gemini TTS, audio tags, эмоции, chunk limits |
| По ситуации | [docs/12-gemini-prompting-templates.md](docs/12-gemini-prompting-templates.md) | Нужны project-native Gemini examples, prompt templates, QA checklist |
| Примеры | [examples/](examples/) | Нужен образец сценария, .env.example, Remotion-поток или двухголосый подкаст |

## Обязательный быстрый алгоритм

1. **Безопасность прежде всего.** Прочитай `docs/03-security-and-secrets.md`.
   Создай `.env.example` и `.env` из шаблона, убедись в `.gitignore`, попроси ключи ОДИН раз.
2. **Bootstrap проекта.** Создай болванки: `script.md` (если нет), `out/`,
   `.env.example` уже создан. Если CLI не установлен — поставь Python/UV/FFmpeg
   (если среда позволяет), затем выбери сборку по `docs/02-install.md`.
3. **Выбор провайдера.** Если пользователь не указал — прочитай
   `docs/05-providers-and-models.md`, предложи варианты. По умолчанию:
   `polza-chat-audio` с `openai/gpt-audio-mini` (дёшево, рубли) или
   `polza-tts` с `openai/gpt-4o-mini-tts` (классический TTS, ~1.07 ₽/мин).
   Локальные `audio.cpp` routes не выбирать неявно: сначала прочитать
   `docs/14-local-audio-cpp-models.md`, проверить модель, platform route и GPU.
4. **Проверка окружения.** `voiceover doctor --provider <X> --with-timings [--timing-provider <Y>] --json`.
   Убедись что `workflow_ok: true`.
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
   Если тайминги нужны отдельно — используй ДРУГОЙ `--output-dir`/`--run-id`,
   не перезаписывай папку платного прогона:
   `voiceover timings --audio "out/prod/<full>.mp3" --timing-provider <X> --output-dir "out" --run-id "prod-timings" --json`.
8. **Статус/артефакты.** `voiceover status --run-id "prod" --json`; прочитай `manifest.json`, `run_state.json`, `generation.log`.
   В receipt проверь `execution_source`: source kind, revision/dirty и package-tree SHA-256.

## Security-first правила

- **НИКОГДА не читай `.env`.** Даже чтобы проверить наличие ключа.
- **НИКОГДА не проси пользователя прислать ключ в чат.**
- Создай `.env.example` с placeholder-ами `pza_...` и `sk-or-v1-...`.
- Создай `.env` из `.env.example` сам. Попроси пользователя ОДИН раз вписать ключи в `.env`.
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
| Создаёт .env.example, .gitignore, script.md, out/ — все болванки проекта | Читает .env или значения ключей |
| Проверяет окружение через doctor | Конфигурирует системный PATH |
| Валидирует Markdown-сценарий | Выдумывает несвязанный творческий контент |
| Генерирует озвучку через любой из 5 провайдеров с provider-specific retry, safe rerun и manifest/log | Рендерит Remotion-видео |
| Извлекает тайминги через локальный faster-whisper ИЛИ облачные OpenRouter/Groq/xAI Whisper | Правит исходники voiceover-pipeline |
| Читает manifest.json → артефакты | Использует words-per-second при наличии timings |
| Объясняет провайдеров, модели, голоса, цены (7 TTS + 6 STT моделей) | Гарантирует будущие цены провайдеров |
| Диагностирует ошибки по exit codes | Правит исходники voiceover-pipeline |

## Режим F: Two-speaker podcast (dialogue)

Когда пользователь просит «подкаст», «диалог», «два ведущих» или
«вопрос-ответ» — маршрутизируй на канонический `format: dialogue`.
`gemini-dialogue` остаётся совместимым alias; manifest и state всегда пишут
`dialogue`. Для OpenRouter нужен `openrouter-tts` +
`google/gemini-3.1-flash-tts-preview`; локальный путь требует явного
`omnivoice-local` и admitted voice bank.

> OpenRouter поддерживает один top-level `voice` на запрос. Диалоговый
> генератор делает **один запрос на реплику (turn)**, выбирает голос по alias,
> не отправляет недокументированный `multi_speaker_voice_config`, удаляет
> `Alias:` из synthesized text и сохраняет детерминированные паузы. `input`
> byte-equals текущему turn text: style/profile/vibe/соседний контекст не
> отправляются. Перед concat обязателен явный `--tts-quality-provider`.

- Агент — автор сценария: когда пользователь просит готовый подкаст или
  озвучку, агент может написать структурированный диалоговый скрипт
  (`podcast.md`) из темы и состава ведущих пользователя. Это не
  «выдумывание контента»: сценарий нужен для производства запрошенного
  артефакта.
- Ровно два различных спикера. Для OpenRouter голоса берутся из Gemini voice
  list; для OmniVoice нужны два profile ID с разными `reference_sha256`.
- Выбор провайдера остаётся явным и подтверждается пользователем до
  платного вызова.
- OmniVoice admission/runtime происходит один раз, но каждая реплика вызывает
  свой bound profile. Обычный одноголосый OmniVoice по-прежнему один native
  session на прогон.
- Offline contract проверен, но audible acceptance OpenRouter и OmniVoice
  остаётся отдельным human gate: не заявляй фактическое слышимое различие
  голосов или выпуск 0.6.0 без двух PASS.
- Полный workflow: `docs/08-workflows.md` → «Agent Podcast Workflow»;
  формат: `docs/04-input-format.md`; пример: `examples/gemini-dialogue-podcast.md`.

## Чеклист готового навыка

- [ ] `SKILL.md` ≤300 строк, docs/ ≤300, examples/ ≤300
- [ ] Каталог ведёт в реальные файлы
- [ ] `description` покрывает реальные фразы пользователя (русский + English keywords)
- [ ] Есть trigger checks: should trigger / should not trigger / boundary
- [ ] Есть smoke tests: минимум 8 кейсов с assertions
- [ ] Есть regression set
- [ ] Security-first правила на первом месте
- [ ] Все команды — bare (`voiceover ...`), кроме секции разработки
- [ ] Цены и модели — только тестированные, с реальных прогонов
- [ ] Локальный TTS-сценарий не содержит необработанных цифр и machine-readable ID
- [ ] Скорость/качество локальных моделей привязаны к конкретному receipt/corpus, а static и live evidence не смешаны
- [ ] Навык не привязан к одному агенту или IDE
- [ ] `docs/00-version-log.md` содержит совместимость с версией CLI
