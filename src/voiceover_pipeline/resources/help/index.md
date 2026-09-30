---
topic: index
title: Темы справки voiceover-pipeline
summary: Оглавление атомарных тем help — каждая тема это один короткий Markdown-файл в установленном пакете.
related: start.quick, speech.simple, speech.parts, speech.legacy, runs.resume, asr.transcribe, history.find, history.costs, search.lexical, search.semantic, cli.json, providers.polza
---

`voiceover help` читает Markdown из установленного пакета. Рабочий каталог, репозиторий, ключи, FFmpeg, GPU и сеть для справки не нужны.

## Как читать справку

```bash
voiceover help                      # эта страница (тема по умолчанию: index)
voiceover help start.quick          # человекочитаемый вывод с заголовком
voiceover help speech.parts --raw   # чистый Markdown без ANSI
voiceover help search.semantic --json
```

- `--raw` печатает ровно тот Markdown, который лежит в пакете, без frontmatter и без ANSI.
- `--json` печатает один объект: `topic`, `title`, `summary`, `related`, `markdown`.
- Неизвестная или недопустимая тема — usage error с exit `2`; с `--json` это тот же безопасный error envelope.
- Обычный `--help` остаётся коротким: он перечисляет команды, а не темы.

## Темы

| Тема | О чём |
|---|---|
| `start.quick` | Установка, где ключи и где данные, первые команды |
| `speech.simple` | Одна короткая реплика: `--text`, голос, `--vibe` |
| `speech.parts` | Явные части через `format: speech-parts`, подсчёт размера |
| `speech.legacy` | Прежние Markdown и dialogue-сценарии, совместимость |
| `runs.resume` | `--resume`, `status`, `history resume`/`sync`, когда нужен новый прогон |
| `asr.transcribe` | Локальное распознавание и реальные ограничения таймингов |
| `history.find` | История, импорт старых `out/`, доступность файлов |
| `history.costs` | Exact/unknown/local-попытки, валюты и полнота учёта |
| `search.lexical` | Офлайн-поиск FTS5 по сохранённым текстам |
| `search.semantic` | Почему semantic/hybrid сейчас недоступны (DEFERRED) |
| `cli.json` | stdout/stderr, `--json`, события, exit codes |
| `providers.polza` | Настройка Polza и подтверждённые маршруты |

Платные команды в темах не выполняются справкой. Любой live/paid запрос — отдельное решение владельца.
