# OpenRouter + Google Gemini TTS

Google Gemini 3.1 Flash TTS Preview — чистый text-to-speech через OpenRouter. Не chat-модель, а классический `/audio/speech`.

## Модель

**`google/gemini-3.1-flash-tts-preview`**

- Тип: `text → speech`
- Контекст: 32 000 токенов
- Языки: 70+, включая русский
- Выдаёт: PCM 24 kHz / 16-bit mono
- Фишки: 200+ встроенных аудио-тегов (через Google AI Studio), два диктора с независимыми голосами, авто-watermark SynthID
- Цена: ~$0.03/min

> **Образец:** [openrouter-gemini-tts-chunk-01.ogg](openrouter-gemini-tts-chunk-01.ogg) ($0.03/мин) — первый чанк в OGG Vorbis 42 kbps mono, голос `Puck`.

## Голоса

| Голос | Характер |
|---|---|
| `Puck` | Спокойный, вдумчивый мужской (**дефолт**) |
| `Charon` | Резонансный мужской |
| `Fenrir` | Смелый мужской |
| `Orus` | Мужской |
| `Aoede` | Женский |
| `Kore` | Женский |
| `Zephyr` | Эфирный |

Все голоса — от Google. `alloy`/`echo`/`onyx` (OpenAI-голоса) здесь **не работают**.

## Style prompt

В отличие от Polza, где system prompt жёстко фиксирован, здесь можно управлять подачей через стилевой prompt. Он вставляется перед текстом в поле `input` — и модель читает текст **этим** голосом.

**Дефолтный:**

```text
Голос технического подкаста: спокойный, вдумчивый, живой и уверенный.
Тёплый мужской тембр, средний темп, ясная артикуляция, без театральности.
```

**Свой:**

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --style-prompt "Энергичный голос ведущего новостей: громкий, быстрый."
```

**Fallback:** если основной prompt отвергнут с ошибкой `No successful provider responses`, автоматически пробуется укороченный:

```text
Спокойный живой голос подкаста. Тёплый мужской тембр, средний темп,
вдумчивая подача.
```

## Запуск

```powershell
# Стандартный прогон
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Puck"

# Другим голосом
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Charon"

# Со своим стилем
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --style-prompt "Тёплый голос рассказателя: медленный, доверительный, с паузами."
```

## Ключ

Нужен `OPENROUTER_API_KEY` в `.env`. `.env` ищется в CWD и вверх по родительским директориям.

```env
OPENROUTER_API_KEY=sk-or-v1-твой-ключ
```

## Как работает

Запрос уходит на `https://openrouter.ai/api/v1/audio/speech`:

```json
{
  "model": "google/gemini-3.1-flash-tts-preview",
  "input": "<style prompt>\n\n<текст чанка>",
  "voice": "Puck",
  "response_format": "pcm"
}
```

**Важно:** Gemini через OpenRouter принимает только `response_format="pcm"`. `"mp3"` выдаст ошибку 400. Пайплайн запрашивает PCM и сам конвертирует в MP3 через FFmpeg.

Ответ — raw PCM bytes. Пайплайн пишет их в MP3-файл.

## Цены

Цена с реального прогона (12 чанков, тестовый сценарий):

| Модель | Общая стоимость | Длина | Цена минуты |
|---|---:|---:|---|
| `google/gemini-3.1-flash-tts-preview` | $0.312752 | 620.0 сек | $0.03026632/min |

Pricing snapshot от OpenRouter (`GET /api/v1/models?output_modalities=speech`):

| Модель | prompt | completion |
|---|---|---|
| `google/gemini-3.1-flash-tts-preview` | 0.000001 | 0.00002 |

Точная стоимость берётся из `GET /api/v1/generation?id={generation_id}` → поле `total_cost`. Пайплайн делает до 4 попыток с задержкой 3 сек, потому что OpenRouter не всегда сразу отдаёт usage.
