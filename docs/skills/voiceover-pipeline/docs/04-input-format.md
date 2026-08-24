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
- **Числа в локальном TTS:** до генерации обязательно преобразовать цифры,
  даты, проценты, размеры, версии и десятичные дроби в естественные слова с
  нужным падежом. Например, писать «тридцать три целых одна десятая процента»,
  а не `33,1%`. ID, хэши и пути либо объяснять человечески, либо исключать из
  произносимого текста. Валидатор помечает цифры как warning, но для локальной
  production-озвучки агент обязан устранить warning до запуска.
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
- Если есть только `warnings` — для облачного TTS решение зависит от модели и
  сценария. Для локального TTS warning `contains_digits` сначала исправить;
  запускать production-озвучку с необработанными цифрами нельзя.

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

Для production безопаснее разделять стадии:

```powershell
voiceover generate --script "script.md" --run-id "prod" --json --resume
voiceover timings --audio "out/prod/prod-voiceover-model.mp3" --run-id "prod" --json --overwrite
```

Если всё же используется `generate --with-timings`, зависимость `faster-whisper`
проверяется до первого платного TTS-запроса.

## Dialogue format

`dialogue` — provider-neutral формат для подкастов с двумя дикторами,
голосами и inline emotion tags. `gemini-dialogue` принимается как
compatibility alias, но planner, manifest и run state используют только
`dialogue`.

> OpenRouter поддерживает один top-level `voice` на запрос. Диалоговый
> маршрут делает один запрос на реплику с голосом её alias, не отправляет
> `multi_speaker_voice_config` и не озвучивает `Alias:`.

```markdown
---
format: dialogue
language: ru
model: google/gemini-3.1-flash-tts-preview
speakers:
  Host:
    display_name: Ведущая
    voice: Kore
    profile: warm, confident technical host
  Guest:
    display_name: Гость
    voice: Puck
    profile: calm, thoughtful technical expert
vibe: >
  Russian technical podcast. Natural question-and-answer conversation.
allowed_tags:
  - warmly
  - curious
  - serious
  - short pause
max_chunk_bytes: 3500
---

Host: [warmly] Что умеет утилита?
Guest: Она создаёт озвучку, тайминги и субтитры.

******

Host: [curious] Можно работать полностью локально?
Guest: Да. Для одноголосой озвучки доступны OmniVoice и Qwen.
```

Правила:

- Frontmatter читает pipeline, в TTS он не отправляется как реплика.
- Ровно два aliases, alphanumeric без пробелов. Один или три спикера, а также
  повтор голоса/profile — ошибка (`SPEAKER_COUNT_INVALID`,
  `DUPLICATE_SPEAKER_VOICE`).
- Для OpenRouter голос берётся из Gemini voice list. Для OmniVoice `voice` —
  profile ID из admitted bank; два профиля должны иметь разные
  `reference_sha256`.
- Emotion tags пишутся по-английски в квадратных скобках. Каждый смысловой
  блок проверяется по UTF-8 bytes: default `3500`, hard limit `4000`.
- `******` делит диалог на смысловые блоки. Каждая реплика становится turn:
  один provider request с одним документированным voice, 250 ms между turns,
  600 ms после разделителя и 0 ms после последнего turn.
- После trim паузы материализуются как локальный PCM silence в final audio.
  Receipt каждого turn содержит `turn_index`, `speech_duration_ms`,
  `audio_sha256`, offsets, voice и доступный fingerprint; transcript и
  reference text не публикуются.
- `--resume` требует state с точным `synthesis_identity`. Изменение каста,
  fingerprint, текста, порядка, паузы или trim policy, а также orphan turn
  audio без state, завершается exit `30` до provider call.

Валидация и OpenRouter generation:

```powershell
voiceover validate --script "script.md" --format dialogue --agent --json
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --format dialogue `
  --script "script.md" `
  --run-id "podcast-prod" `
  --json `
  --resume
```

Для локального варианта укажи `--provider omnivoice-local --mode preset
--voice-bank <catalog.json>` и явно выбранные два profile ID в frontmatter.
Проверенный offline contract не заменяет отдельное human listening acceptance:
не заявляй два слышимо разных голоса до PASS для нужного provider.

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
- `provider` или `service` — `polza-chat-audio`, `polza-tts`, `openrouter-tts`, `qwen-local`, `omnivoice-local`.
- `model` — модель выбранного провайдера.
- `voice` — голос, если провайдер его поддерживает.
- `fallback_voice` — только для `polza-chat-audio`.
- `style_prompt` или `prompt` — работает для OpenRouter Gemini TTS, для остальных режимов валидатор даст warning.
- `max_chunk_chars` — настроенный лимит символов на чанк; для `omnivoice-local` с его штатной моделью валидатор показывает локальный лимит 420, если CLI-лимит не задан.

JSON-отчёт также содержит `spoken_text` с политикой raw digits и информационной статистикой состава текста: количеством Latin-символов и Latin-слов, смешанных по алфавиту слов, всех букв и долей Latin. Для OmniVoice raw digits блокируют валидацию; для остальных маршрутов это warning.

Валидация:

```powershell
voiceover validate --script "script.md" --format voiceover --agent --json
```

Генерация может использовать metadata без повторения provider/model/voice:

```powershell
voiceover generate --script "script.md" --run-id "prod" --json --resume
```

CLI-флаги переопределяют frontmatter, например:

```powershell
voiceover generate `
  --script "script.md" `
  --provider openrouter-tts `
  --model "openai/gpt-4o-mini-tts-2025-12-15" `
  --voice alloy
```
