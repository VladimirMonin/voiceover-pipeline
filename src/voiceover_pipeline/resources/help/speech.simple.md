---
topic: speech.simple
title: Одна короткая реплика
summary: generate --text создаёт ровно одну часть; голос выбирается явно, а непустой vibe допустим только на маршруте с отдельным полем instructions (Gemini 3.8).
related: start.quick, speech.parts, providers.polza, runs.resume
---

## Что делает `--text`

`generate --text "..."` собирает ровно одну часть: одно произносимое предложение, один голос, один запрос провайдеру. Следующий вызов **платный**, а не проверка справки; запуск возможен только после отдельного разрешения владельца.

```bash
voiceover generate --provider polza-chat-audio --text "Добрый вечер." --voice ash --run-id greeting --json
```

Правила, которые применяются до любого запроса:

- `--text` и явный `--script` взаимоисключающие: это один вход.
- `--text` не принимает `--format`: короткая реплика всегда одна часть.
- Без `--provider` берётся default приложения (`polza-chat-audio`, модель `openai/gpt-audio-mini`, голос `ash`). Другие провайдеры требуют явного `--voice`, если у их модели нет голоса по умолчанию.
- `--vibe` — свободная строка-инструкция, а не текст для чтения. Она не попадает в произносимый текст и хранится в снимке прогона отдельно.

## `--vibe` на Gemini 3.8-маршрутах

Непустой `--vibe` передаётся провайдеру только на маршруте, который несёт
направление отдельным полем `instructions` — Gemini 3.8 Flash и Flash-Lite на
`polza-tts`/`openrouter-tts`. На любом другом маршруте он отклоняется **до**
чтения ключа и любого POST:

```json
{"status": "error", "error": "BLOCKED_PROVIDER_CONTRACT: ...", "code": 30,
 "details": {"error_code": "BLOCKED_PROVIDER_CONTRACT"}}
```

`google/gemini-3.8-flash-tts` и `google/gemini-3.8-flash-lite-tts` — обычные
admitted-модели обоих облачных провайдеров, без experimental opt-in. Один
`--text`/часть = один scalar `voice` на POST; произносимый `input` — ровно
`--text`, а эффективная инструкция уходит отдельным полем `instructions`, которое
для Gemini **не документировано**. Эффект режиссуры не гарантирован.
Датированные наблюдения — `docs/reports/2026-10-01-s06-gemini38-live-probes.md` в
репозитории; новые GET/POST только с разрешением и проверяемым бюджетом.

Реплика без vibe допускается на зарегистрированном маршруте провайдера; это не доказывает его live-доступность или качество звука.

## Что получается

```text
out/greeting/
├── manifest.json
├── run_state.json
├── <run-id>-voiceover-<model>.mp3
└── chunks/chunk_01.mp3
```

`--audio-format wav` меняет контейнер только итогового merged-файла и допускается лишь на DB-first нативном маршруте; промежуточные части в `chunks/` остаются MP3.
