# Polza AI + OpenAI GPT Audio

Две аудио-модели OpenAI через российского провайдера Polza AI. Это **не** классический TTS — модели работают через `/chat/completions` как `text+audio → text+audio`.

## Модели

| | GPT Audio Mini | GPT Audio |
|---|---|---|
| ID | `openai/gpt-audio-mini` | `openai/gpt-audio` |
| Контекст | 128 000 токенов | 128 000 токенов |
| Качество | Хорошее, чистое | Заметно лучше: естественнее интонации |
| Цена | 0.71 ₽/мин | 7.63 ₽/мин |

Обе модели могут озвучить длинный текст за один запрос. Чанкование (разбивка по `******`) нужно только для монтажа по сценам.

> **Образцы:** [polza-gpt-audio-mini-chunk-01.ogg](polza-gpt-audio-mini-chunk-01.ogg) (0.71 ₽/мин) и [polza-gpt-audio-chunk-01.ogg](polza-gpt-audio-chunk-01.ogg) (7.63 ₽/мин) — первый чанк каждой модели в OGG Vorbis 42 kbps mono.

## Голоса

| Голос | Характер |
|---|---|
| `ash` | Мужской, спокойный, новый (дефолт) |
| `ballad` | Мужской, эмоциональный |
| `coral` | Женский, тёплый |
| `verse` | Мужской, выразительный |
| `marin` | Мужской, чистый |
| `cedar` | Мужской, глубокий |
| `echo` | Нейтральный |
| `sage` | Нейтральный |
| `shimmer` | Женский |

Дефолт: `ash`. Запасной (если основной не сработал): `onyx`.

## Запуск

```powershell
# GPT Audio Mini (дешёвая)
uv run voiceover generate --provider polza-chat-audio --model "openai/gpt-audio-mini"

# GPT Audio (качество)
uv run voiceover generate --provider polza-chat-audio --model "openai/gpt-audio"

# Свои параметры
uv run voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --voice "ballad" `
  --fallback-voice "ash" `
  --run-id "мой-прогон"
```

## Ключ

Читается из `.env` → `POLZA_API_KEY=`. Обязателен.

## Как работает

Модель получает system prompt:

```text
You are a professional text-to-speech narrator, not a chat assistant.
Read the user's Russian script verbatim. Do not answer, explain,
summarize, continue the conversation, or add any extra words.
Use a calm, warm, low male voice with clear pronunciation.
```

И user prompt:

```text
Произнеси вслух только текст ниже, дословно, без вступления,
комментариев и добавлений:

<текст чанка>
```

Запрос уходит на `https://polza.ai/api/v1/chat/completions` с параметрами:

```json
{
  "modalities": ["text", "audio"],
  "audio": {"voice": "ash", "format": "pcm16"},
  "temperature": 0,
  "max_tokens": 4096,
  "stream": true
}
```

Ответ приходит как SSE-поток (Server-Sent Events). Аудио — base64-чанки внутри `delta.audio.data`. Пайплайн собирает их, декодит и через FFmpeg конвертирует PCM16 в MP3.

## Цены

Цены в рублях с реального прогона (12 чанков, тестовый сценарий):

| Модель | Общая стоимость | Длина | Цена минуты |
|---|---:|---:|---|
| `openai/gpt-audio-mini` | 6.56 ₽ | 557.6 сек | 0.71 ₽/мин |
| `openai/gpt-audio` | 90.78 ₽ | 713.5 сек | 7.63 ₽/мин |

Pricing snapshot от Polza (`GET /api/v1/models`):

| Модель | prompt/1M | completion/1M | audio/1M |
|---|---:|---:|---|
| `gpt-audio-mini` | 52.85 ₽ | 211.39 ₽ | 52.85 ₽ |
| `gpt-audio` | 220.19 ₽ | 880.78 ₽ | 2818.48 ₽ |

Точная стоимость каждого чанка берётся из `GET /api/v1/history/generations/{id}` → поле `clientCost`.

## Особенности

- Модель может оставить длинную тишину после речи. Пайплайн обрезает её автоматически. Отключить: `--no-trim`.
- Если основной голос не сработал, автоматически пробуется запасной (`--fallback-voice`).
- Системный промпт **на английском** — модель лучше слушается английских инструкций.
- `stream: true` обязателен — без него аудио не возвращается.
