---
topic: speech.legacy
title: Прежние форматы сценария
summary: Markdown, voiceover-metadata и dialogue остаются совместимыми; что именно они умеют и где их граница.
related: speech.parts, speech.simple, runs.resume
---

Старые форматы не удалены: они по-прежнему валидируются и генерируются. Новые входы (`--text`, `speech-parts`) сосуществуют с ними, а не заменяют их молча.

## `markdown` (по умолчанию)

Обычный текст, части разделяются строкой с разделителем `******` (переопределяется `--delimiter`).

```markdown
Первый фрагмент озвучки.

******

Второй фрагмент озвучки.
```

```bash
voiceover split --script script.md --json      # id и размеры чанков
voiceover validate --script script.md --json
```

## `format: voiceover`

Markdown с frontmatter metadata: провайдер, модель и голос заданы в самом сценарии, поэтому прогон воспроизводим.

```markdown
---
format: voiceover
provider: polza-tts
model: openai/gpt-4o-mini-tts
voice: ash
max_chunk_chars: 2000
---

Первый фрагмент озвучки.

******

Второй фрагмент озвучки.
```

## `format: dialogue`

Двухголосый сценарий: карта спикеров (`speakers` с `display_name`, `voice`, `profile`), `language`, `model`, `vibe`, `allowed_tags`, `max_chunk_bytes`. `gemini-dialogue` остаётся compatibility alias с идентичным планом, а state/manifests всегда пишут канонические `format: dialogue`.

```bash
voiceover validate --script podcast.md --format dialogue --agent --json
voiceover generate --help
```

Второй вызов показывает лишь параметры, **не** запускает генерацию. Для платного OpenRouter Gemini dialogue-исполнения требуется `--tts-quality-provider` с заранее установленной локальной проверкой (`qwen-local` или `nemotron-local`) либо отдельно разрешённым облачным `xai-stt`. Нельзя выдавать `--resume` за новый прогон или опускать обязательный gate.

## Границы

- `--vibe` работает только с `--text`; в legacy-формате он отклоняется с `VIBE_UNSUPPORTED_FORMAT` (exit `2`) до чтения ключа и POST, потому что формат не умеет нести инструкцию.
- Per-part голоса бывают только на подтверждённом маршруте, который шлёт один cast-voice на запрос (`openrouter-tts` + `google/gemini-3.1-flash-tts-preview`). Провайдер с одним голосом на прогон не подменяет голоса молча.
- Диалоговые маршруты, требующие облачной ASR-проверки качества (`--tts-quality-provider xai-stt`), относятся только к двум admitted dialogue-маршрутам; остальные остаются на legacy-исполнителе.
- Слышимое различие голосов и live-приёмка остаются ручными гейтами: офлайн-контракт не доказывает звук.
