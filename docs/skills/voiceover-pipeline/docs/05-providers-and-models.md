# Провайдеры, модели, голоса и цены

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: 4 провайдера, 7 протестированных моделей, все голоса, реальные цены.
> Это главный справочник для выбора провайдера/модели/голоса.

## Обзор

voiceover-pipeline поддерживает четыре TTS-провайдера с единым интерфейсом:

| Провайдер | Тип | API | Валюта | Ключ | Provider ID |
|---|---|---|---|---|---|
| Polza Chat Audio | Cloud, chat-based | `/chat/completions` | RUB | `POLZA_API_KEY` | `polza-chat-audio` |
| Polza TTS | Cloud, TTS + ElevenLabs | `/audio/speech`, `/media` | RUB | `POLZA_API_KEY` | `polza-tts` |
| OpenRouter TTS | Cloud, агрегатор | `/audio/speech` | USD | `OPENROUTER_API_KEY` | `openrouter-tts` |
| Qwen-local | Local GPU | Внутрипроцессный | Бесплатно | Не нужен | `qwen-local` |

---

## Polza Chat Audio — OpenAI GPT Audio

Через `/chat/completions` как `text+audio → text+audio`. **Не классический TTS** —
модель ведёт диалог голосом, может добавить речь.

### Модели

| Модель | ID | Качество | Цена/мин | Примечание |
|---|---|---|---|---|
| GPT Audio Mini | `openai/gpt-audio-mini` | Хорошее, чистое | **~0.004 RUB** | anomalous benchmark, модель добавила речь |
| GPT Audio | `openai/gpt-audio` | Заметно лучше, естественные интонации | **~7.00 RUB** | Самый качественный из Polza Chat Audio |

### Голоса

| Голос | Пол | Характер |
|---|---|---|
| `ash` | М | Спокойный (**дефолт**) |
| `ballad` | М | Эмоциональный |
| `coral` | Ж | Тёплый |
| `verse` | М | Выразительный |
| `marin` | М | Чистый |
| `cedar` | М | Глубокий |
| `echo` | — | Нейтральный |
| `sage` | — | Нейтральный |
| `shimmer` | Ж | — |
| `onyx` | — | **Запасной** (fallback при ошибке основного) |

### Особенности

- System prompt на английском — модель лучше слушается
- Stream SSE — аудио base64-чанками, пайплайн собирает и конвертирует
- Обрезка тишины после речи (отключить: `--no-trim`)
- Fallback voice: если основной голос не сработал → `--fallback-voice` (default `onyx`)
- Точная стоимость: `GET /api/v1/history/generations/{id}` → `clientCost`

---

## Polza TTS — OpenAI TTS + ElevenLabs

Model-aware dispatch: `openai/*` → `/audio/speech`, `elevenlabs/*` → `/media`.

### Модели OpenAI TTS через Polza

| Модель | ID | Цена/мин | Endpoint |
|---|---|---|---|
| GPT-4o Mini TTS | `openai/gpt-4o-mini-tts` | **~1.07 RUB** | `POST /api/v1/audio/speech` |

Ответ: `{"audio":"<base64>","contentType":"audio/mpeg","usage":{"cost_rub":...}}`

### Модели ElevenLabs через Polza

| Модель | ID | Цена/мин | Endpoint |
|---|---|---|---|
| ElevenLabs Turbo | `elevenlabs/text-to-speech-turbo-2-5` | **~3.51 RUB** | `POST /api/v1/media` |
| ElevenLabs Multilingual | `elevenlabs/text-to-speech-multilingual-v2` | **~7.57 RUB** | `POST /api/v1/media` |

Запрос `/media`: `{"model":"...","input":{"prompt":"...","voice":"Rachel","language_code":"ru"},"async":true}`
→ poll `GET /media/{id}` → download MP3 с `data[0].url`.

### Голоса OpenAI TTS (Polza TTS + OpenRouter)

| Голос | Пол | Характер |
|---|---|---|
| `alloy` | — | Нейтральный |
| `ash` | М | Спокойный |
| `ballad` | М | Эмоциональный |
| `coral` | Ж | Тёплый |
| `echo` | — | Нейтральный |
| `fable` | — | Британский |
| `nova` | Ж | Мягкий |
| `onyx` | М | Глубокий |
| `sage` | — | Нейтральный |
| `shimmer` | Ж | — |
| `verse` | М | Выразительный |

Все 11 голосов доступны в `polza-tts` и `openrouter-tts` (OpenAI-модели).
**Дефолт:** `alloy` для Polza TTS и OpenRouter OpenAI TTS.

### Голоса ElevenLabs через Polza (21 имя)

`Rachel` (Ж, тёплый, **дефолт**), `Aria`, `Roger`, `Sarah`, `Laura`, `Charlie`, `George`, `Callum`, `River`, `Liam`, `Charlotte`, `Alice`, `Matilda`, `Will`, `Jessica`, `Eric`, `Chris`, `Brian`, `Daniel`, `Lily`, `Bill`.

Это Polza display-names из их allowlist, не native ElevenLabs `voice_id`.

### Особенности Polza TTS

- **OpenAI TTS:** `--voice alloy` (дефолт), ответ — JSON с base64 MP3
- **ElevenLabs:** `--voice Rachel` (дефолт), async `/media` — submit → poll (до 5 мин) → download
- Единый `POLZA_API_KEY` для обоих polza-провайдеров
- Style prompt НЕ используется для Polza TTS (не поддерживается endpoint)

---

## OpenRouter TTS — Gemini + OpenAI TTS

Агрегатор, единый `/audio/speech`. Две модели, разные семейства голосов.

### Модели

| Модель | ID | Цена/мин | Style prompt | Голоса |
|---|---|---|---|---|
| Gemini TTS | `google/gemini-3.1-flash-tts-preview` | **~$0.030** | Да | Google (30) |
| OpenAI Mini TTS | `openai/gpt-4o-mini-tts-2025-12-15` | **~$0.00041** | Нет (пропускается) | OpenAI (11) |

### Голоса Gemini TTS (30 имён)

**Дефолт:** `Puck` (М, спокойный, вдумчивый).

`Puck`, `Charon`, `Fenrir`, `Orus`, `Aoede`, `Kore`, `Zephyr`, `Leda`, `Callirrhoe`, `Autonoe`, `Enceladus`, `Iapetus`, `Umbriel`, `Algieba`, `Despina`, `Erinome`, `Algenib`, `Rasalgethi`, `Laomedeia`, `Achernar`, `Alnilam`, `Schedar`, `Gacrux`, `Pulcherrima`, `Achird`, `Zubenelgenubi`, `Vindemiatrix`, `Sadachbia`, `Sadaltager`, `Sulafat`.

### Style prompt (Gemini)

Управление подачей через `--style-prompt`:

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --style-prompt "Энергичный голос ведущего: громкий, быстрый, чёткий."
```

Дефолт: «Голос технического подкаста: спокойный, вдумчивый, живой и уверенный.»
Fallback: если style prompt отвергнут, автоматически пробуется укороченный вариант.

**OpenAI TTS модели НЕ принимают style_prompt** — CLI автоматически пропускает.

### Особенности OpenRouter

- Только `response_format="pcm"` (не mp3). Пайплайн сам конвертирует PCM → MP3.
- Ретраи для цены: `GET /api/v1/generation?id=...` до 4 попыток с паузой 3 сек.
- Cost может быть `null` если OpenRouter не успел обновить usage.

---

## Qwen3-TTS (локальный, бесплатный)

Open-source модель синтеза речи. Работает локально на GPU (NVIDIA, CUDA).

### Модели

| Модель | HF ID | Режим |
|---|---|---|
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 24 preset-голоса |
| Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Клонирование голоса |

### Голоса (preset, 24 шт)

`Aiden` (М, спокойный, **дефолт**), `Alina`, `Amelia`, `Arthur`, `Callum`, `Carter`, `Elijah`, `Ethan`, `Evelyn`, `Isabella`, `Jack`, `James`, `Landon`, `Liam`, `Lily`, `Lucas`, `Mason`, `Mia`, `Natalia`, `Olivia`, `Paul`, `Sofia`, `Theo`, `Violet`.

### Требования

- NVIDIA GPU + CUDA (~4 GB VRAM)
- Модель ~3.4 GB, скачивается один раз
- Extras: `voiceover-pipeline[voiceover-qwen]`

---

## Быстрый выбор

| Задача | Провайдер | Модель | Цена |
|---|---|---|---|
| Самый дешёвый, рубли | Polza Chat Audio | `openai/gpt-audio-mini` | ~0.004 RUB/мин |
| Классический TTS, рубли | Polza TTS | `openai/gpt-4o-mini-tts` | ~1.07 RUB/мин |
| Чистый голос, рубли | Polza TTS | `elevenlabs/text-to-speech-turbo-2-5` | ~3.51 RUB/мин |
| Лучшее качество речи, рубли | Polza TTS | `elevenlabs/text-to-speech-multilingual-v2` | ~7.57 RUB/мин |
| Качество интонаций (chat) | Polza Chat Audio | `openai/gpt-audio` | ~7.00 RUB/мин |
| Самый дешёвый TTS, доллары | OpenRouter | `openai/gpt-4o-mini-tts-2025-12-15` | ~$0.00041/мин |
| Западные голоса, качество | OpenRouter | `google/gemini-3.1-flash-tts-preview` | ~$0.030/мин |
| Бесплатно, есть GPU | Qwen-local | CustomVoice (preset) | Бесплатно |

Цены — реальные smoke-прогоны 2026-04-29, не гарантия провайдера.
Актуальный список всегда в `docs/00-version-log.md`.
