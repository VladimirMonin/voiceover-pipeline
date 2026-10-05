---
topic: speech.parts
title: Явные части: format speech-parts
summary: Строгий YAML с отдельным голосом и текстом на часть, бюджет символов до запроса и отсутствие автоматической нарезки.
related: speech.simple, speech.legacy, runs.resume, providers.polza
---

## Зачем явные части

`format: speech-parts` — единственный вход, где автор сам задаёт упорядоченный список частей, у каждой ровно один голос и ровно один произносимый текст. Приложение не режет текст само: нет скрытого чанкера, нет «догадки по абзацам».
Успешный заказ подкаста — **один сценарный YAML, один run и один итоговый MP3**.
На маршруте, подтверждённом для per-part голосов, смена голоса означает
отдельный запрос на часть, а не один multi-speaker POST. По одной только теме
сначала нужен полный произносимый сценарий; TTS не сочиняет подкаст.
Пример без vibe может пройти route preflight; платная генерация требует
разрешения владельца и доказанной верхней границы стоимости. Gemini 3.8 Flash
и Flash-Lite — обычные admitted-модели без opt-in.

```yaml
version: 1
format: speech-parts
parts:
  - voice: ash
    text: Первая часть.
  - voice: ash
    text: Вторая часть.
```

Правила формата:

- ключи top-level: `version`, `format`, `vibe`, `parts`; у части: `voice`, `text`, `vibe`;
- неизвестный ключ, дубликат ключа, вложенный mapping и tab-отступ отклоняются, а не игнорируются; BOM принимается;
- `provider`/`model` в документе не дублируются — только CLI или постоянные defaults;
- `--voice`, `--vibe` и `--style-prompt` вместе с `speech-parts` отклоняются вместо скрытого override: все голоса и инструкции живут в YAML.

Если добавить общий или частный непустой `vibe`, эффективная инструкция части = общий vibe, пустая строка, vibe части. Она никогда не смешивается с произносимым `text`. Такой `vibe` допустим только на маршруте, который несёт направление отдельным полем (Gemini 3.8 Flash/Flash-Lite на `polza-tts`/`openrouter-tts`); любой другой маршрут отказывает до ключа и POST (см. «Маршрут»).

## Проверка до денег

```bash
voiceover validate --script parts.yaml --format speech-parts --provider polza-chat-audio --json
```

`validate --json` возвращает `parts`, `request_chars`, `route.admitted` и `route.reason`. Бюджет считается как `len(text) + len(effective_vibe) + len(required_text_wrapper) <= 5000` для **каждой** части; это приложение-side консервативная политика, а не доказанный предел провайдера. Поздняя over-limit часть останавливает прогон до первого POST (`SPEECH_PART_TOO_LONG`, exit `2`).

## Маршрут

`speech-parts` идёт только через DB-first нативную генерацию, без legacy fallback. Непустой `vibe` и модель без подтверждённой per-part-схемы fail-closed с `BLOCKED_PROVIDER_CONTRACT` (exit `30`) до чтения ключа и POST.

Gemini 3.8 Flash (`google/gemini-3.8-flash-tts`) и Flash-Lite
(`google/gemini-3.8-flash-lite-tts`) — **обычные admitted-модели** обоих
облачных speech-провайдеров (`polza-tts`, `openrouter-tts`), без experimental
opt-in:

- один scalar `voice` на POST, то есть **отдельный запрос на часть**, а не
  multi-speaker POST;
- эффективная инструкция уходит отдельным полем `instructions`, которое для Gemini
  **не документировано**, а произносимый `input` — ровно `text`;
- эффект режиссуры и внешний счёт не гарантируются;
- наблюдённый WAV (при запросе MP3) распознаётся и конвертируется, а приватное
  тело ответа и receipt сохраняются до разбора: `history resume`/`sync`
  переигрывают то же наблюдение локально без второго POST.

Устаревшее compatibility-написание `--allow-experimental-gemini-speech-parts`
принимается и записывается в снимок, но больше ничего не требует и ничего не
отклоняет.

Разные голоса на разные части умеет также маршрут
`openrouter-tts` + `google/gemini-3.1-flash-tts-preview`. Любой другой маршрут
говорит одним голосом на прогон и не может честно озвучить разные голоса:
приложение отказывает, а не подменяет голос.

`history resume`/`sync` восстанавливают прогон из снимка и не перечитывают исходный YAML.
