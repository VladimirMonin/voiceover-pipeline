# Входной формат: Markdown-сценарий

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: формат Markdown-сценария, разделитель, валидация, рекомендации.

## Формат сценария

Сценарий — это Markdown-файл, в котором чанки (сцены, абзацы, блоки текста)
разделены строкой `******` (шесть звёздочек).

```markdown
Первый фрагмент сценария. Может быть несколько предложений.
Озвучивается как один чанк — один MP3-файл.

******

Второй фрагмент сценария. Это отдельный чанк.
Его длина не зависит от других.

******

Третий фрагмент сценария.
```

## Правила

- Разделитель по умолчанию: `******` (можно изменить через `--delimiter`)
- Каждый НЕПУСТОЙ блок между разделителями становится отдельным чанком
- Пустые блоки (пробелы/переносы между разделителями) игнорируются
- Файл читается как UTF-8 или UTF-8 with BOM
- Чанки именуются `chunk_01`, `chunk_02`, `chunk_03`, ...
- Если чанков 0 — ошибка exit code 2

## Рекомендации по чанкам

- **Для Remotion:** один чанк = одна сцена. Так проще сопоставлять
  `chunks.json` с Remotion-композициями.
- **Размер:** до 2000 символов на чанк. Валидатор предупредит если больше.
- **Числа в тексте:** TTS может прочитать их не так, как вы ожидаете
  (например «123» как «сто двадцать три» вместо «один-два-три»).
  Валидатор пометит чанки с цифрами как warning.
- **Спецсимволы:** Markdown-разметка (**жирный**, *курсив*) игнорируется
  TTS-моделями. Пишите чистый текст для озвучки.

## Валидация сценария

Перед генерацией всегда запускай валидатор:

```powershell
voiceover validate --script "script.md" --json
```

JSON-ответ:

```json
{
  "status": "success",
  "valid": true,
  "chunks": 5,
  "total_chars": 1240,
  "issues": [],
  "warnings": [
    {"chunk": 3, "type": "contains_digits"}
  ]
}
```

**Поведение агента при issues:**

- Если `valid: false` (есть `issues`) — НЕ запускай генерацию. Покажи пользователю
  проблемные чанки и предложи исправить.
- Если есть только `warnings` — предупреди пользователя, но генерацию можно
  запускать.

## Просмотр чанков без генерации

```powershell
voiceover split --script "script.md" --json
```

JSON-ответ:

```json
{
  "status": "success",
  "chunks": [
    {"id": "chunk_01", "chars": 245},
    {"id": "chunk_02", "chars": 180},
    {"id": "chunk_03", "chars": 312}
  ]
}
```

Полезно чтобы понять, как сценарий разобьётся, не тратя деньги на генерацию.

## Gemini Dialogue format

Для Gemini 3.1 Flash TTS через OpenRouter есть отдельный формат для подкастов
с двумя дикторами, голосами и inline emotion tags:

```markdown
---
format: gemini-dialogue
language: ru
model: google/gemini-3.1-flash-tts-preview
speakers:
  Speaker1:
    display_name: Первый диктор
    voice: Puck
    profile: calm, warm, confident technical host
  Speaker2:
    display_name: Второй диктор
    voice: Kore
    profile: curious, energetic co-host
vibe: >
  Russian technical podcast. Warm, smart, conversational.
allowed_tags:
  - warmly
  - curious
  - laughs
  - serious
  - short pause
max_chunk_bytes: 3500
---

Speaker1: [warmly] Привет. Это начало выпуска.
Speaker2: [curious] А это второй голос с другим характером.

******

Speaker1: [serious] Новый смысловой чанк.
Speaker2: [laughs] Короткая эмоциональная реакция.
```

Правила:

- Frontmatter читает pipeline, в TTS он не отправляется как реплика.
- В dialogue body разрешены только aliases из `speakers`, например `Speaker1:`.
- Speaker aliases должны быть alphanumeric без пробелов.
- Голоса должны быть из списка Gemini voices.
- Emotion tags пишутся по-английски в квадратных скобках.
- Каждый чанк проверяется по UTF-8 bytes. Default safety limit: `3500`, hard limit: `4000`.
- Если хотя бы один чанк невалиден, `generate --format gemini-dialogue` не стартует.

Валидация:

```powershell
voiceover validate `
  --script "script.md" `
  --format gemini-dialogue `
  --model "google/gemini-3.1-flash-tts-preview" `
  --agent `
  --json
```

Генерация:

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --format gemini-dialogue `
  --script "script.md" `
  --run-id "podcast-prod" `
  --with-timings `
  --json `
  --overwrite
```

## Voiceover metadata format

Для обычных single-speaker режимов можно хранить provider/model/voice прямо в
Markdown frontmatter. Plain Markdown без frontmatter остаётся совместимым.

```markdown
---
format: voiceover
provider: polza-tts
model: openai/gpt-4o-mini-tts
voice: ash
max_chunk_chars: 2000
---

Первый фрагмент обычной озвучки.

******

Второй фрагмент обычной озвучки.
```

Поддерживаемые поля:

- `format: voiceover` — включает metadata validator.
- `provider` или `service` — `polza-chat-audio`, `polza-tts`, `openrouter-tts`, `qwen-local`.
- `model` — модель выбранного провайдера.
- `voice` — голос, если провайдер его поддерживает.
- `fallback_voice` — только для `polza-chat-audio`.
- `style_prompt` или `prompt` — работает для OpenRouter Gemini TTS, для остальных режимов валидатор даст warning.
- `max_chunk_chars` — лимит символов на чанк.

Валидация:

```powershell
voiceover validate --script "script.md" --format voiceover --agent --json
```

Генерация может использовать metadata без повторения provider/model/voice:

```powershell
voiceover generate --script "script.md" --run-id "prod" --json --overwrite
```

CLI-флаги переопределяют frontmatter, например:

```powershell
voiceover generate `
  --script "script.md" `
  --provider openrouter-tts `
  --model "openai/gpt-4o-mini-tts-2025-12-15" `
  --voice alloy
```
