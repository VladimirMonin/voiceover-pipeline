# Polza TTS

Три модели text-to-speech через российского провайдера Polza AI:

- `openai/gpt-4o-mini-tts` — через `/api/v1/audio/speech` (OpenAI-compatible endpoint)
- `elevenlabs/text-to-speech-turbo-2-5` — через `/api/v1/media` (Polza Media API)
- `elevenlabs/text-to-speech-multilingual-v2` — через `/api/v1/media` (Polza Media API)

## Модели

| | `openai/gpt-4o-mini-tts` | `elevenlabs/text-to-speech-turbo-2-5` | `elevenlabs/text-to-speech-multilingual-v2` |
|---|---|---|---|
| Endpoint | `/audio/speech` | `/media` | `/media` |
| Цена/мин | ~1.07 ₽ | ~3.51 ₽ | ~7.57 ₽ |
| Цена snapshot | 52.65 ₽/1M prompt + 1053 ₽/1M completion | 4500 ₽/1M chars | 9000 ₽/1M chars |
| Контекст | до 4096 токенов | до 5000 символов | до 5000 символов |

Цены из реальных smoke-прогонов (2 чанка, тестовый сценарий). Snapshot — из `GET /api/v1/models`.

> **Образцы:**
> - [polza-openai-gpt-4o-mini-tts-chunk-01.ogg](polza-openai-gpt-4o-mini-tts-chunk-01.ogg) — GPT-4o Mini TTS, голос `ash`, ~1.07 ₽/мин
> - [polza-elevenlabs-turbo-2-5-chunk-01.ogg](polza-elevenlabs-turbo-2-5-chunk-01.ogg) — ElevenLabs Turbo 2.5, голос `Rachel`, ~3.51 ₽/мин
> - [polza-elevenlabs-multilingual-v2-chunk-01.ogg](polza-elevenlabs-multilingual-v2-chunk-01.ogg) — ElevenLabs Multilingual v2, голос `Rachel`, ~7.57 ₽/мин

## Голоса

### OpenAI TTS voices (для `/audio/speech`)

| Голос | Характер |
|---|---|
| `alloy` | Нейтральный, универсальный (**дефолт**) |
| `ash` | Мужской, спокойный |
| `ballad` | Мужской, эмоциональный |
| `coral` | Женский, тёплый |
| `echo` | Нейтральный |
| `fable` | Британский, выразительный |
| `nova` | Женский, мягкий |
| `onyx` | Мужской, глубокий |
| `sage` | Нейтральный |
| `shimmer` | Женский, лёгкий |
| `verse` | Мужской, выразительный |

### ElevenLabs voices (для `/media`) — через Polza wrapper names

| Голос |
|---|
| `Rachel` (**дефолт**) |
| `Aria` |
| `Roger` |
| `Sarah` |
| `Laura` |
| `Charlie` |
| `George` |
| `Callum` |
| `River` |
| `Liam` |
| `Charlotte` |
| `Alice` |
| `Matilda` |
| `Will` |
| `Jessica` |
| `Eric` |
| `Chris` |
| `Brian` |
| `Daniel` |
| `Lily` |
| `Bill` |

> Важно: Polza использует **display-имена** из своего allowlist, а не native ElevenLabs `voice_id`. Голоса вроде `21m00Tcm4TlvDq8ikWAM` не подойдут.

## Запуск

```powershell
# OpenAI TTS через Polza (дефолтная модель polza-tts)
voiceover generate --provider polza-tts --model "openai/gpt-4o-mini-tts" --voice "ash"

# ElevenLabs Turbo 2.5
voiceover generate `
  --provider polza-tts `
  --model "elevenlabs/text-to-speech-turbo-2-5" `
  --voice "Rachel" `
  --run-id "my-run"

# ElevenLabs Multilingual v2
voiceover generate `
  --provider polza-tts `
  --model "elevenlabs/text-to-speech-multilingual-v2" `
  --voice "Aria" `
  --run-id "my-run"
```

## Ключ

Используется `POLZA_API_KEY` из `.env`. Тот же ключ, что и для `polza-chat-audio`.

```env
POLZA_API_KEY=pza_...
```

## Как работает

### OpenAI TTS: `/api/v1/audio/speech`

```json
// Request
{
  "model": "openai/gpt-4o-mini-tts",
  "input": "текст чанка",
  "voice": "ash",
  "response_format": "mp3"
}

// Response — JSON с base64
{
  "audio": "<base64>",
  "contentType": "audio/mpeg",
  "usage": {
    "characters": 254,
    "cost_rub": 0.40787955,
    "cost": 0.40787955
  }
}
```

Формат: MP3, моно, 24 kHz. Пайплайн декодирует base64 и записывает напрямую, без FFmpeg.

### ElevenLabs TTS: `/api/v1/media` (async)

```json
// Request — POST, возвращает task с id
{
  "model": "elevenlabs/text-to-speech-turbo-2-5",
  "input": {
    "prompt": "текст чанка",
    "voice": "Rachel",
    "language_code": "ru"
  },
  "async": true
}

// Response (submit)
{
  "id": "gen_...",
  "status": "pending"
}

// Poll GET /api/v1/media/{id} до completed
{
  "id": "gen_...",
  "status": "completed",
  "data": [
    {"url": "https://s3.polza.ai/..."}
  ],
  "usage": {
    "cost_rub": 0.1575,
    "cost": 0.1575
  }
}
```

Пайплайн опрашивает статус каждые 5 сек (до 300 сек total), затем скачивает MP3 по URL из `data[0].url`.

## Цены

Цены с реальных smoke-прогонов (2 чанка, тестовый сценарий):

| Модель | Стоимость | Длина | Цена/мин |
|---|---:|---:|---|
| `openai/gpt-4o-mini-tts` | 0.59 ₽ | 33.4 сек | 1.07 ₽/мин |
| `elevenlabs/text-to-speech-turbo-2-5` | 1.67 ₽ | 28.5 сек | 3.51 ₽/мин |
| `elevenlabs/text-to-speech-multilingual-v2` | 3.33 ₽ | 26.4 сек | 7.57 ₽/мин |

Pricing snapshot от Polza (`GET /api/v1/models`):

| Модель | prompt/1M | completion/1M | tts/1M chars |
|---:|---:|---:|---|
| `gpt-4o-mini-tts` | 52.65 ₽ | 1053.00 ₽ | — |
| `elevenlabs/text-to-speech-turbo-2-5` | — | — | 4500 ₽ |
| `elevenlabs/text-to-speech-multilingual-v2` | — | — | 9000 ₽ |

Точная стоимость берётся из `usage.cost_rub` в ответе API или из `GET /api/v1/history/generations/{id}` → `clientCost`.
