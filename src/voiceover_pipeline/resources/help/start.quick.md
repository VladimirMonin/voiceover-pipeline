---
topic: start.quick
title: Быстрый старт
summary: Установка, расположение ключей и данных, офлайн-команды и явная граница платной генерации.
related: speech.simple, runs.resume, providers.polza, cli.json
---

## Установка

`voiceover` и `voiceover-pipeline` — один и тот же CLI.

```bash
uvx voiceover-pipeline doctor --json      # запуск без установки
pipx install voiceover-pipeline           # постоянная установка
uv pip install "voiceover-pipeline[timing-whisper]"   # плюс локальные тайминги
```

## Где ключи

Ключ читается приложением при проверке/вызове провайдера; `doctor` тоже читает его для проверки наличия, но `help` не читает ключ или `.env`. Значение ключа не печатается:

1. переменная окружения текущего процесса — приоритет;
2. файл `.env` в текущем рабочем каталоге (CWD).

Имена переменных: `POLZA_API_KEY`, `OPENROUTER_API_KEY` (озвучка), `GROQ_API_KEY`, `X_AI_API_KEY` (облачные тайминги/ASR). Поиска `.env` по родительским каталогам нет: значение берётся из окружения или из `<CWD>/.env`.

`doctor` проверяет наличие ключей и показывает путь к проверяемому `.env` в поле `env_file.path`, но не печатает значения; запускайте его только в одобренном окружении. Для `help`, `validate`, `history list/show/costs`, `search --mode lexical`, `list` и `index` ключ не нужен. `history resume` может сделать новый платный POST, а `history sync` — GET по известному remote id; они **не** входят в эту офлайн-группу.

## Где данные

| Что | Расположение |
|---|---|
| Сценарий по умолчанию | `./in/` |
| Артефакты прогона | `./out/<run-id>/` (переопределяется `--output-dir`) |
| История (SQLite) | `VOICEOVER_HOME` или платформенный data-каталог пользователя |
| Постоянные несекретные настройки | `./settings.toml` (необязательный) |

`VOICEOVER_HOME` должен быть абсолютным путём; относительное значение — ошибка. История живёт отдельно от `out/` и не привязана к текущему каталогу.

## Первые офлайн-команды

```bash
voiceover help --json
voiceover validate --script in/script.md --json           # проверка сценария без POST
voiceover list providers --json
voiceover history list --json
voiceover generate --help                                 # только аргументы, не генерация
```

`doctor` работает без сети, но для проверки наличия ключа **читает** текущий `.env`; его нужно запускать только в одобренном окружении. `status`, `history list/show/costs`, `search --mode lexical`, `index` и `help` не делают запросов к провайдеру. `generate` потенциально платен и требует отдельного решения владельца; не копируйте его из справки как офлайн-проверку.

Подробные маршруты — `voiceover help providers.polza`, короткая реплика — `voiceover help speech.simple`.
