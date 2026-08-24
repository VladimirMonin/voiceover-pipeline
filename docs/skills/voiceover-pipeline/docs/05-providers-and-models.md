# Провайдеры, модели, голоса и цены

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: облачные и локальные TTS-провайдеры, голоса и реальные цены.
> Это главный справочник для выбора провайдера/модели/голоса.

## Обзор

voiceover-pipeline поддерживает облачные TTS-провайдеры и две локальные
модельные линии. Подробности нового hybrid runtime и его benchmark boundaries:
[`docs/14-local-audio-cpp-models.md`](14-local-audio-cpp-models.md).

| Провайдер | Тип | API | Валюта | Ключ | Provider ID |
|---|---|---|---|---|---|
| Polza Chat Audio | Cloud, chat-based | `/chat/completions` | RUB | `POLZA_API_KEY` | `polza-chat-audio` |
| Polza TTS | Cloud, TTS + ElevenLabs | `/audio/speech`, `/media` | RUB | `POLZA_API_KEY` | `polza-tts` |
| OpenRouter TTS | Cloud, агрегатор | `/audio/speech` | USD | `OPENROUTER_API_KEY` | `openrouter-tts` |
| Qwen-local | Local GPU | Внутрипроцессный | Бесплатно | Не нужен | `qwen-local` |
| OmniVoice local | Local GPU | `audio.cpp` | Бесплатно, CC-BY-NC | Не нужен | `omnivoice-local` |

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

## OpenRouter TTS — Gemini

Агрегатор, единый `/audio/speech`. Текущий speech-каталог допускает Gemini.

### Модели

| Модель | ID | Цена/мин | Style prompt | Голоса |
|---|---|---|---|---|
| Gemini TTS | `google/gemini-3.1-flash-tts-preview` | **~$0.030** | Нет | Google (30) |

### Голоса Gemini TTS (30 имён)

**Дефолт:** `Puck` (М, спокойный, вдумчивый).

`Puck`, `Charon`, `Fenrir`, `Orus`, `Aoede`, `Kore`, `Zephyr`, `Leda`, `Callirrhoe`, `Autonoe`, `Enceladus`, `Iapetus`, `Umbriel`, `Algieba`, `Despina`, `Erinome`, `Algenib`, `Rasalgethi`, `Laomedeia`, `Achernar`, `Alnilam`, `Schedar`, `Gacrux`, `Pulcherrima`, `Achird`, `Zubenelgenubi`, `Vindemiatrix`, `Sadachbia`, `Sadaltager`, `Sulafat`.

### Verbatim input (Gemini)

OpenRouter `/audio/speech` получает только точный произносимый текст текущей
реплики. `style_prompt`, `vibe`, speaker `profile`, labels и соседние реплики
не добавляются к `input`; отдельное поле `prompt` также не отправляется.
Подача выбирается только top-level полем `voice`. CLI отклоняет явные
`--style-prompt` и `--style-prompt-file` до платного запроса.

OpenRouter делает ровно один платный synthesis-запрос на turn; автоматического
generation retry или fallback-запроса с другим prompt нет.

### Особенности OpenRouter

- Gemini-запрос содержит `model`, `input`, `voice`, `response_format="pcm"`;
  ответ — raw audio body. JSON, data URI, base64 field и SSE отклоняются.
- OpenRouter dialogue требует явный `--tts-quality-provider`: каждый turn
  транскрибируется и строго сверяется до final concat; receipt не хранит текст.
- Отсутствующий в текущем speech-каталоге model ID отклоняется до billing.
- `openai/gpt-audio-mini` и `openai/gpt-audio` используют chat-audio контракт,
  а не `/audio/speech`, поэтому не добавляются как ложная замена.
- Ретраи для цены: `GET /api/v1/generation?id=...` до 4 попыток с паузой 3 сек.
- Cost может быть `null` если OpenRouter не успел обновить usage.

---

## Qwen3-TTS (локальный, бесплатный)

Open-source модель синтеза речи. Работает локально на GPU (NVIDIA, CUDA).

### Модели

| Модель | HF ID | Режим |
|---|---|---|
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 preset-голосов |
| Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Клонирование голоса |

### Голоса (preset, 9 шт)

`Aiden` (М, американский, спокойный, **дефолт**), `Dylan` (М, пекинский, молодой), `Eric` (М, сычуаньский, живой), `Ono_Anna` (Ж, японский, игривый), `Ryan` (М, английский, динамичный), `Serena` (Ж, тёплый), `Sohee` (Ж, корейский, эмоциональный), `Uncle_Fu` (М, низкий, seasoned), `Vivian` (Ж, яркий, молодой).

### Требования

- NVIDIA GPU + CUDA (~4 GB VRAM)
- Модель ~3.4 GB, скачивается один раз
- Extras: `voiceover-pipeline[voiceover-qwen]`

---

## Распознавание речи и тайминги

Подробный справочник вынесен в [`docs/13-speech-recognition-providers.md`](13-speech-recognition-providers.md):
локальные Faster-Whisper, Qwen3-ASR и Nemotron, облачные OpenRouter Whisper,
Groq Whisper и xAI STT, их модели, виды таймкодов и ограничения.

---

## Быстрый выбор

| Задача | Провайдер | Модель | Цена |
|---|---|---|---|
| Самый дешёвый, рубли | Polza Chat Audio | `openai/gpt-audio-mini` | ~0.004 RUB/мин |
| Классический TTS, рубли | Polza TTS | `openai/gpt-4o-mini-tts` | ~1.07 RUB/мин |
| Чистый голос, рубли | Polza TTS | `elevenlabs/text-to-speech-turbo-2-5` | ~3.51 RUB/мин |
| Лучшее качество речи, рубли | Polza TTS | `elevenlabs/text-to-speech-multilingual-v2` | ~7.57 RUB/мин |
| Качество интонаций (chat) | Polza Chat Audio | `openai/gpt-audio` | ~7.00 RUB/мин |
| Западные голоса, качество | OpenRouter | `google/gemini-3.1-flash-tts-preview` | ~$0.030/мин |
| Бесплатно, есть GPU | Qwen-local | CustomVoice (preset) | Бесплатно |
| Локальные тайминги, сегменты | faster-whisper | `small` | Бесплатно (CPU) |
| Облачные тайминги + сегменты | groq-whisper | `whisper-large-v3-turbo` | $0.04/час |
| Облачные тайминги, слова + conf | xai-stt | `grok-stt` | xAI pricing |
| Только текст, облачно | openrouter-whisper | `openai/whisper-1` | ~$0.002/мин |

Цены — реальные smoke-прогоны 2026-04-29 (TTS) / 2026-05-27 (STT), не гарантия провайдера.
Актуальный список всегда в `docs/00-version-log.md`.