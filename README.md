# Voiceover Pipeline

CLI для генерации озвучки (TTS) из Markdown/YAML-сценариев и получения
таймингов/субтитров. Машинный контракт для агентов: `--json`, семантические
exit codes, `manifest.json` в каталоге прогона.

> **Статус.** Версия приложения в `pyproject.toml` — `0.6.1`. Обозначение
> `v0.7.0` из плана развития — ещё не опубликованный целевой релиз. Всё ниже
> описывает состояние репозитория, а не гарантию опубликованного пакета.

## Возможности

- Облачные TTS: Polza GPT Audio (`/chat/completions`), Polza TTS
  (`/audio/speech`, `/media`), OpenRouter Gemini TTS.
- Локальные TTS: Qwen3-TTS (GPU) и OmniVoice.
- Двухспикерный подкаст (`format: dialogue`) и явные части
  (`format: speech-parts`).
- Тайминги/субтитры: локальный `faster-whisper` либо облачные
  `groq-whisper`/`xai-stt`; локальное распознавание `qwen-local`/`nemotron-local`.
- Локальная SQLite-история и офлайн лексический FTS5-поиск по сохранённым текстам.
- Упакованная атомарная справка: `voiceover help [TOPIC] [--raw | --json]`.

## Установка из исходников

`git clone`, установка пакетов и загрузка моделей используют сеть: агент выполняет
их только после отдельного разрешения владельца и в согласованном окружении.

```bash
git clone https://github.com/VladimirMonin/voiceover-pipeline
cd voiceover-pipeline
uv sync --group dev                    # CLI + dev-инструменты (lint, mypy, pytest)
uv sync --extra timing-whisper         # + локальный faster-whisper
uv run voiceover doctor --json
```

Console scripts `voiceover` и `voiceover-pipeline` эквивалентны.

### Extras

| Extra | Добавляет |
|---|---|
| `timing-whisper` | `faster-whisper` + `ctranslate2` |
| `voiceover-qwen` | Qwen3-TTS (`torch`, `qwen-tts`, `soundfile`, `numpy`) |
| `asr-qwen` | `qwen-asr` |
| `asr-nemotron` | Nemotron ASR (`accelerate`, `librosa`, `torch`, `transformers`) |
| `cuda` | CUDA-библиотеки для Windows/Linux |

`asr-qwen`, `voiceover-qwen` и `asr-nemotron` **взаимоисключающие** (см.
`[tool.uv] conflicts` в `pyproject.toml`): ставьте только тот локальный маршрут,
который действительно нужен. Единого «all extras» набора нет.

## Ключи и безопасность

Реальные ключи пользователь хранит в приватном env-файле вне инструментов
агента (или в уже существующих переменных окружения процесса). Агент никогда не
создаёт, не копирует и не читает реальный `.env`; приложение может прочитать его
только при обращении к ключу (включая `doctor`). В репозитории есть только шаблон
[`.env.example`](docs/skills/voiceover-pipeline/examples/env-example.md) с
синтетическими placeholder-ами. `.env` не коммитится.

Порядок разрешения значения ключа при обращении команды к секрету:

1. непустая переменная окружения процесса — приоритет; содержимое файла тогда не читается;
2. явный глобальный `voiceover --env-file PATH <command> ...` (проверяется по метаданным как regular file);
3. `<call-time CWD>/.env` — для совместимости.

Поиска `.env` по родительским каталогам нет. Явный `--env-file` **заменяет**
CWD-файл, а не дополняет его: фоллбэка на `<CWD>/.env` не происходит.
Отсутствующий, не-regular или недоступный для проверки метаданных явный путь
fail-closed с exit `20`, даже если в окружении есть пригодный ключ.

Read-only справка не читает ключ и env-файл:

```bash
voiceover help                       # справка из установленного пакета, без ключей и .env
voiceover help start.quick --json
```

`doctor` отдельно проверяет наличие ключей и сообщает разрешённый путь в
`checks.env_file.path`, но никогда — значение. При непригодном явном `--env-file`
он завершается с exit `0` и `status: "success"`, однако `workflow_ok: false`.
Запускайте `doctor` только в одобренном окружении.

## Быстрый старт

Офлайн-команды ничего не отправляют:

```bash
voiceover validate --script in/script.md --json
voiceover list providers --json
voiceover history list --json
voiceover search "текст" --mode lexical --json
```

Генерация потенциально платная и требует отдельного разрешения владельца:

```bash
voiceover generate \
  --provider polza-chat-audio \
  --model openai/gpt-audio-mini \
  --script in/script.md \
  --run-id prod \
  --json \
  --resume
```

`--resume` продолжает прерванный прогон без пересоздания готовых частей;
платные данные не перезаписываются без явного `--confirm-delete-paid-audio`.
Интегрированные локальные тайминги (`generate --with-timings`) требуют заранее
закешированной модели: она проверяется до платного POST и не докачивается неявно.
Отдельный `timings --timing-provider faster-whisper` может скачать модель Whisper
при первом использовании — это сетевая операция с разрешения владельца.

## Артефакты

```
out/<run-id>/
├── manifest.json                          ← entry-point для агентов
├── run_state.json                         ← resumable state после каждой части
├── generation.log                         ← человекочитаемый лог
├── <run-id>-voiceover-<model>.mp3         ← итоговый MP3
├── <run-id>.timings.json                  ← тайминги (ms)
├── <run-id>.srt                           ← субтитры SRT
└── chunks/
    ├── chunk_01.mp3 ... chunk_NN.mp3
    └── chunks.json
```

Необязательный `<CWD>/settings.toml` хранит несекретные настройки
(`[history] enabled`, `[search] default_mode`, пути локальных ASR-моделей).
Режим `search` по умолчанию — `lexical`; настроенные `semantic`/`hybrid`
честно отказывают (`SEARCH_MODE_DEFERRED`), пока семантический поиск вынесен
в отдельный план следующего релиза.

## Документация

- [Индекс документации](docs/README.md) — машинный контракт, планы, отчёты, skill.
- [Agent CLI Contract](docs/agent-cli-contract.md) — JSON/exit codes, stdout/stderr, границы безопасности.
- [Agent Skill](docs/skills/voiceover-pipeline/SKILL.md) — установка и озвучка агентом.
- [Agent Development Workflow](doc/agent-workflow.md) — dirty tree, Kanban, Git/release approvals.
- Установленная справка: `voiceover help`, `voiceover help start.quick`, `voiceover help search.semantic`.

## Разработка

После отдельно согласованной установки зависимостей проверки можно запускать
без сети:

```bash
uv run --offline --frozen pytest
uv run --offline --frozen ruff check src tests
uv run --offline --frozen ruff format --check src tests
uv run --offline --frozen mypy --no-incremental
```

Тесты детерминированные и офлайн: temp-каталоги, fixtures и моки; без реальных
ключей, платных API и внешних загрузок.

## Legal

Независимая обёртка над сторонними провайдерами, не аффилирована с OpenAI,
Google, OpenRouter, Polza.ai, Qwen или CTranslate2/faster-whisper. Условия
использования сгенерированного аудио определяет выбранный провайдер. Не
загружайте приватные голосовые образцы и сгенерированную речь без разрешения.

## License

MIT. См. [LICENSE](LICENSE).
