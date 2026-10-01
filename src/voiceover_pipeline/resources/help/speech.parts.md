---
topic: speech.parts
title: Явные части: format speech-parts
summary: Строгий YAML с отдельным голосом и текстом на часть, бюджет символов до запроса и отсутствие автоматической нарезки.
related: speech.simple, speech.legacy, runs.resume, providers.polza
---

## Зачем явные части

`format: speech-parts` — единственный вход, где автор сам задаёт упорядоченный список частей, у каждой ровно один голос и ровно один произносимый текст. Приложение не режет текст само: нет скрытого чанкера, нет «догадки по абзацам».
Пример без vibe может пройти route preflight; платная генерация требует отдельного
разрешения владельца и доказанной верхней границы стоимости.

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

Если добавить общий или частный непустой `vibe`, эффективная инструкция части = общий vibe, пустая строка, vibe части. Она никогда не смешивается с произносимым `text`, но текущие provider-маршруты откажут до ключа и POST (см. «Маршрут» ниже).

## Проверка до денег

```bash
voiceover validate --script parts.yaml --format speech-parts --provider polza-chat-audio --json
```

`validate --json` возвращает `parts`, `request_chars`, `route.admitted` и `route.reason`. Бюджет считается как `len(text) + len(effective_vibe) + len(required_text_wrapper) <= 5000` для **каждой** части; это приложение-side консервативная политика, а не доказанный предел провайдера. Поздняя over-limit часть останавливает прогон до первого POST (`SPEECH_PART_TOO_LONG`, exit `2`).

## Маршрут

`speech-parts` идёт только через DB-first нативную генерацию, без legacy fallback. Непустая vibe и неверифицированная candidate-модель сейчас fail-closed с `BLOCKED_PROVIDER_CONTRACT` (exit `30`) до чтения ключа и POST. Разные голоса на разные части умеет лишь уже подтверждённый маршрут `openrouter-tts` + `google/gemini-3.1-flash-tts-preview`: провайдер, который говорит одним голосом на прогон, не может честно озвучить разные голоса, и приложение отказывает, а не подменяет голос.

`history resume`/`sync` восстанавливают прогон из снимка и не перечитывают исходный YAML.
