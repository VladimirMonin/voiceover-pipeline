# Провайдеры, модели и голоса

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: выбор класса TTS-маршрута и его ограничений; это не каталог текущих тарифов.
> Каноническая справка — `voiceover help providers.polza`, текущие регистрации — `voiceover list`.

## Обзор

voiceover-pipeline поддерживает облачные TTS-провайдеры и две локальные
модельные линии. Подробности нового hybrid runtime и его benchmark boundaries:
[`docs/14-local-audio-cpp-models.md`](14-local-audio-cpp-models.md).

> Зарегистрированные модели и голоса смотрите в `voiceover list providers --json` /
> `voiceover list voices --provider <X> --json`, а маршрут и ключи — в
> `voiceover help providers.polza`. Ни одна из этих команд не подтверждает
> доступность, качество или тариф внешнего провайдера. Исторические цены остаются
> в отчётах, а не в инструкции по выбору: до любого provider GET/POST нужны
> отдельное разрешение владельца и доказанная верхняя граница платного вызова.

| Провайдер | Тип | API | Валюта | Ключ | Provider ID |
|---|---|---|---|---|---|
| Polza Chat Audio | Cloud, chat-based | `/chat/completions` | RUB | `POLZA_API_KEY` | `polza-chat-audio` |
| Polza TTS | Cloud, TTS + ElevenLabs | `/audio/speech`, `/media` | RUB | `POLZA_API_KEY` | `polza-tts` |
| OpenRouter TTS | Cloud, агрегатор | `/audio/speech` | USD | `OPENROUTER_API_KEY` | `openrouter-tts` |
| Qwen-local | Local GPU | Внутрипроцессный | Нет тарифа провайдера | Не нужен | `qwen-local` |
| OmniVoice local | Local GPU | `audio.cpp` | Нет тарифа провайдера, CC-BY-NC | Не нужен | `omnivoice-local` |

---

## Polza Chat Audio — OpenAI GPT Audio

Через `/chat/completions` как `text+audio → text+audio`. **Не классический TTS** —
модель ведёт диалог голосом, может добавить речь.

### Модели

В реестре есть `openai/gpt-audio-mini` и `openai/gpt-audio`. Это chat-audio,
а не гарантия дословного TTS: модель может добавить речь. Фактическая
доступность и стоимость требуют отдельной проверки до платного запроса.

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
| `onyx` | — | Голос; значение `--fallback-voice` по умолчанию (совместимость, без автоматического запроса) |

### Особенности

- System prompt на английском — модель лучше слушается
- Stream SSE — аудио base64-чанками, пайплайн собирает и конвертирует
- Обрезка тишины после речи (отключить: `--no-trim`)
- Один streaming POST на чанк: `--fallback-voice` (default `onyx`) принимается только для совместимости и не отправляет второй запрос при ошибке. Другой голос требует нового явного прогона.
- Наблюдённая стоимость может поступить из `GET /api/v1/history/generations/{id}` → `clientCost`; этот GET является сетью, не выполняйте его без отдельного разрешения владельца.

---

## Polza TTS — OpenAI TTS + ElevenLabs

Model-aware dispatch: `openai/*` → `/audio/speech`, `elevenlabs/*` → `/media`.

### Модели OpenAI TTS через Polza

| Зарегистрированная модель | Endpoint |
|---|---|
| `openai/gpt-4o-mini-tts` | `POST /api/v1/audio/speech` |
| `google/gemini-3.8-flash-tts` | `POST /api/v1/audio/speech` |
| `google/gemini-3.8-flash-lite-tts` | `POST /api/v1/audio/speech` |

Ответ: `{"audio":"<base64>","contentType":"audio/mpeg","usage":{"cost_rub":...}}`

### Модели ElevenLabs через Polza

| Зарегистрированная модель | Endpoint |
|---|---|
| `elevenlabs/text-to-speech-turbo-2-5` | `POST /api/v1/media` |
| `elevenlabs/text-to-speech-multilingual-v2` | `POST /api/v1/media` |

Запрос `/media`: `{"model":"...","input":{"prompt":"...","voice":"Rachel","language_code":"ru"},"async":true}`
→ poll `GET /media/{id}` → download MP3 с `data[0].url`.

### Зарегистрированные голоса OpenAI TTS через Polza

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

Все 11 имён зарегистрированы для OpenAI-модели `polza-tts`; регистрация не
доказывает live-доступность. **Дефолт Polza TTS:** `alloy`. Текущий реестр
`openrouter-tts` содержит только Gemini-модель, не OpenAI TTS: проверяй
`voiceover list providers --json` и `voiceover list voices --provider openrouter-tts --json`
вместо переноса Polza-голосов на другой маршрут.

### Голоса ElevenLabs через Polza

**Дефолт:** `Rachel`. `voiceover list voices --provider polza-tts --json`
возвращает **и** OpenAI-, и ElevenLabs-голоса в общем `voices`; для последних
используй `voice_categories.elevenlabs` в том же JSON. Это Polza display-names
из allowlist, не native ElevenLabs `voice_id`; CLI-регистрация не доказывает
доступность или звучание у внешнего провайдера.

### Особенности Polza TTS

- **OpenAI TTS:** `--voice alloy` (дефолт), ответ — JSON с base64 MP3
- **ElevenLabs:** `--voice Rachel` (дефолт), async `/media` — submit → poll (до 5 мин) → download
- **ElevenLabs resume:** принятый `/media` task ID сохраняется до poll, поэтому после сбоя `--resume` при совпадении provider/model/voice/script и наличии более ранних MP3 доводит ту же часть GET-запросами без второго платного POST; маркер без ID по-прежнему блокирует `--resume`/`--overwrite`
- Единый `POLZA_API_KEY` для обоих polza-провайдеров
- Style prompt НЕ используется для Polza TTS (не поддерживается endpoint).
- Polza Gemini 3.8 Flash (`google/gemini-3.8-flash-tts`) и Flash-Lite
  (`google/gemini-3.8-flash-lite-tts`) — обычные зарегистрированные модели
  `/audio/speech`. Их голоса — Gemini prebuilt voices (как у `openrouter-tts`),
  голос по умолчанию — `Puck`. Каждый запрос несёт ровно один scalar `voice`, а
  направление части уходит отдельным полем `instructions` (для Gemini не документировано,
  поэтому слышимый эффект режиссуры не гарантирован). Датированные наблюдения
  (возврат WAV при запросе MP3, отсутствие документированной multi-speaker-схемы,
  недоказанный внешний счёт) — в отчёте
  `docs/reports/2026-10-01-s06-gemini38-live-probes.md`; они не отменяют обычную
  поддержку. Устаревший `--allow-experimental-gemini-speech-parts` принимается и
  записывается, но больше ничего не требует.

---

## OpenRouter TTS — Gemini

Агрегатор, единый `/audio/speech`. Текущий speech-каталог допускает Gemini.

### Модели

| Зарегистрированная модель | Style prompt | Голоса |
|---|---|---|
| `google/gemini-3.1-flash-tts-preview` | Нет | `voiceover list voices --provider openrouter-tts --json` |
| `google/gemini-3.8-flash-tts` | Нет | `voiceover list voices --provider openrouter-tts --json` |
| `google/gemini-3.8-flash-lite-tts` | Нет | `voiceover list voices --provider openrouter-tts --json` |

### Голоса Gemini TTS

**Дефолт:** `Puck`. Поддерживаемые в CLI голоса —
`voiceover list voices --provider openrouter-tts --json`; фактическое качество
и доступность определяет только разрешённая проверка провайдера.

### Verbatim input (Gemini)

OpenRouter `/audio/speech` получает только точный произносимый текст текущей
реплики. `style_prompt`, `vibe`, speaker `profile`, labels и соседние реплики
не добавляются к `input`; отдельное поле `prompt` также не отправляется.
Подача выбирается top-level полем `voice`; отдельного поля направления у Gemini 3.1
нет, а у Gemini 3.8 направление реплики передаётся дополнительно отдельным полем
`instructions` (в `input` оно не попадает). CLI отклоняет явные
`--style-prompt` и `--style-prompt-file` до платного запроса.

OpenRouter делает ровно один платный synthesis-запрос на turn; автоматического
generation retry или fallback-запроса с другим prompt нет.

### Особенности OpenRouter

- Gemini-запрос содержит `model`, `input`, `voice`, `response_format="pcm"`;
  ответ — raw audio body. JSON, data URI, base64 field и SSE отклоняются;
  для Gemini 3.8 направление части добавляется отдельным `instructions`, а для 3.1
  по-прежнему нет.
- OpenRouter dialogue требует явный `--tts-quality-provider`: каждый turn
  транскрибируется и строго сверяется до final concat; receipt не хранит текст.
- Отсутствующий в текущем speech-каталоге model ID отклоняется до billing.
- `openai/gpt-audio-mini` и `openai/gpt-audio` используют chat-audio контракт,
  а не `/audio/speech`, поэтому не добавляются как ложная замена.
- Наблюдение цены может потребовать `GET /api/v1/generation?id=...`; в рамках агента такой сетевой шаг допускается только после отдельного разрешения владельца, а `null` остаётся неизвестной ценой.
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
- Веса должны быть подготовлены заранее; приложение не скачивает их неявно, а загрузка требует отдельного разрешения на сеть.
- Extras: `voiceover-pipeline[voiceover-qwen]`

---

## Распознавание речи и тайминги

Подробный справочник вынесен в [`docs/13-speech-recognition-providers.md`](13-speech-recognition-providers.md):
локальные Faster-Whisper, Qwen3-ASR и Nemotron, облачные OpenRouter Whisper,
Groq Whisper и xAI STT, их модели, виды таймкодов и ограничения.

---

## Быстрый выбор без обещания цены

- Нужна дословная речь с выбором голоса — сначала изучите зарегистрированный
  `polza-tts` или `openrouter-tts` маршрут; доступность проверьте отдельно.
- Допустима chat-audio генерация, которая может добавить речь, — рассмотрите
  `polza-chat-audio`, но не предполагайте точность текста или цену по старому smoke.
- Есть заранее подготовленные веса и поддерживаемое GPU-окружение — явным
  выбором пользователя доступны Qwen-local и OmniVoice без тарифа API; ресурсы
  машины и права на голоса остаются отдельной ответственностью.
- Тайминги и распознавание — отдельный [ASR-справочник](13-speech-recognition-providers.md).
  `voiceover list timing-providers --json` показывает регистрацию, не стоимость.

Для платного варианта до каждого live-вызова нужен подтверждённый применимый
тариф и доказанный потолок расходов в разрешённом бюджете; прежние smoke-цены
не годятся для такого расчёта.