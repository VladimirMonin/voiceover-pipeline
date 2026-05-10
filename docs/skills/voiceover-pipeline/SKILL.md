---
name: voiceover-pipeline
description: >
  Установка и использование voiceover-pipeline CLI для генерации озвучки
  из Markdown-сценариев, включая frontmatter metadata для provider/model/voice
  и Gemini dialogue scripts, TTS-аудио, Whisper-таймингов, SRT, manifest.json
  и Remotion-ready артефактов. Используй ВСЕГДА, когда пользователь просит
  озвучить текст, создать voiceover, TTS, аудио для видео, подкаста или
  Remotion, получить тайминги, scene durations, subtitles, timestamps,
  whisper timing или проверить voiceover project. Провайдеры: Polza GPT
  Audio, Polza TTS (OpenAI + ElevenLabs), OpenRouter TTS (Gemini + OpenAI),
  Qwen3-TTS (локальный GPU). Агент создаёт project skeleton
  (.env.example, .gitignore, script.md, out/), просит ключи однократно,
  никогда не читает .env. Триггеры: озвучь, voiceover, TTS, тайминги,
  whisper timing, аудио для видео, подкаст, generate audio, timings for
  Remotion, voiceover-pipeline, выбери провайдера, сравни модели TTS,
  голос для озвучки, format: voiceover, gemini-dialogue, --resume,
  status run, concat partial audio.
---
# Voiceover Pipeline — навык агента

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Детали в docs/ — одна тема на файл, читай по необходимости.
> Запись файлов ТОЛЬКО через инструменты редактирования, не через shell.
> **Совместимость:** voiceover-pipeline 0.4.4, skill revision 2026-05-10.
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
| **C: Timings only** | Готовый MP3, нужны SRT/тайминги | timings --audio → .timings.json + .srt |
| **D: Troubleshoot** | Что-то сломалось | doctor --json → exit code → recovery |

## Когда навык должен срабатывать

**Должен:**
- «озвучь этот markdown-сценарий»
- «сделай voiceover для Remotion»
- «сгенерируй аудио и тайминги из скрипта»
- «нужно получить SRT из MP3»
- «поставь voiceover-pipeline и проверь что работает»
- «сделай подкаст из сценария»
- «voiceover generate с таймингами»
- «какие есть провайдеры/модели/голоса для TTS»
- «выбери дешёвую озвучку»
- «сравни качество TTS моделей»

**Не должен:**
- «объясни как работает git tag»
- «напиши сценарий для ролика» (это творческая задача)
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
| По ситуации | [docs/05-providers-and-models.md](docs/05-providers-and-models.md) | Нужно выбрать провайдера/модель/голос |
| По ситуации | [docs/06-commands-and-flags.md](docs/06-commands-and-flags.md) | Нужен полный CLI-справочник |
| По ситуации | [docs/07-artifacts.md](docs/07-artifacts.md) | Нужно понять что на выходе |
| По ситуации | [docs/08-workflows.md](docs/08-workflows.md) | Нужен готовый end-to-end сценарий |
| По ситуации | [docs/09-troubleshooting.md](docs/09-troubleshooting.md) | Что-то пошло не так |
| По ситуации | [docs/10-evaluation.md](docs/10-evaluation.md) | Проверить качество навыка |
| Примеры | [examples/](examples/) | Нужен образец сценария, .env.example, Remotion-поток |

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
4. **Проверка окружения.** `voiceover doctor --provider <X> --with-timings --json`.
   Убедись что `workflow_ok: true`.
5. **Валидация сценария.** `voiceover validate --script "script.md" --json`.
   Если есть issues — покажи пользователю, не запускай генерацию.
6. **Генерация аудио.** `voiceover generate --provider <X> --model <Y> --script "script.md" --run-id "prod" --json --resume`.
   Не используй `--overwrite` для платной генерации; если run оборвался — продолжай через `--resume`.
7. **Тайминги отдельно.** `voiceover timings --audio "out/prod/<full>.mp3" --run-id "prod" --json --overwrite`.
8. **Статус/артефакты.** `voiceover status --run-id "prod" --json`; прочитай `manifest.json`, `run_state.json`, `generation.log`.

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
| Валидирует Markdown-сценарий | Пишет сценарий за пользователя |
| Генерирует озвучку через любой из 4 провайдеров с retry/resume/state/log | Рендерит Remotion-видео |
| Читает manifest.json → артефакты | Использует words-per-second при наличии timings |
| Объясняет провайдеров, модели, голоса, цены (7 моделей) | Гарантирует будущие цены провайдеров |
| Диагностирует ошибки по exit codes | Правит исходники voiceover-pipeline |

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
- [ ] Навык не привязан к одному агенту или IDE
- [ ] `docs/00-version-log.md` содержит совместимость с версией CLI
