# OpenRouter TTS

Две модели text-to-speech через OpenRouter: Google Gemini TTS и OpenAI GPT-4o Mini TTS.

## Модели

| | Google Gemini | OpenAI GPT-4o Mini |
|---|---|---|
| ID | `google/gemini-3.1-flash-tts-preview` | `openai/gpt-4o-mini-tts-2025-12-15` |
| Endpoint | `/api/v1/audio/speech` | `/api/v1/audio/speech` |
| Контекст | 32 000 токенов | 4 096 токенов |
| Языки | 70+, включая русский | 50+, включая русский |
| Цена/мин | ~$0.030 | ~$0.00041 |

## Голоса

### Google Gemini TTS voices (30)

| Голос | Характер | Пол |
|---|---|---|
| `Puck` | Upbeat | M (**дефолт, только Gemini**) |
| `Charon` | Informative | M |
| `Fenrir` | Excitable | M |
| `Kore` | Firm | F |
| `Zephyr` | Bright | F |
| `Leda` | Youthful | F |
| `Orus` | Firm | M |
| `Aoede` | Breezy | F |
| `Callirrhoe` | Easy-going | F |
| `Autonoe` | Bright | F |
| `Enceladus` | Breathy | M |
| `Iapetus` | Clear | M |
| `Umbriel` | Easy-going | M |
| `Algieba` | Smooth | M |
| `Despina` | Smooth | F |
| `Erinome` | Clear | F |
| `Algenib` | Gravelly | M |
| `Rasalgethi` | Informative | M |
| `Laomedeia` | Upbeat | F |
| `Achernar` | Soft | F |
| `Alnilam` | Firm | M |
| `Schedar` | Even | M |
| `Gacrux` | Mature | F |
| `Pulcherrima` | Forward | F |
| `Achird` | Friendly | M |
| `Zubenelgenubi` | Casual | M |
| `Vindemiatrix` | Gentle | F |
| `Sadachbia` | Lively | M |
| `Sadaltager` | Knowledgeable | M |
| `Sulafat` | Warm | F |

Характеры и пол: [Google AI for Developers — Speech Generation](https://ai.google.dev/gemini-api/docs/speech-generation).

### OpenAI GPT-4o Mini TTS voices

| Голос | Характер |
|---|---|
| `alloy` | Нейтральный, универсальный (**дефолт, только OpenAI TTS**) |
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

Клиент автоматически выбирает дефолтный голос по модели: `Puck` для Gemini, `alloy` для OpenAI TTS.

Важно:
- OpenAI voices вроде `alloy`, `ash`, `nova` не работают для Gemini.
- ElevenLabs voices вроде `Rachel`, `Aria` не работают для Gemini.
- Использовать только Gemini prebuilt voice names, в TitleCase.

> **Образцы:**
> - [openrouter-gemini-tts-chunk-01.ogg](openrouter-gemini-tts-chunk-01.ogg) — Gemini, голос `Puck`, ~$0.030/мин
> - [openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg](openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg) — GPT-4o Mini TTS, голос `ash`, ~$0.00041/мин

## Style prompt (Gemini)

Gemini поддерживает стилевой prompt — передаётся отдельным полем `prompt` в request body (native mode):

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Kore" `
  --style-prompt "Энергичный голос ведущего новостей: громкий, быстрый."
```

### Флаги для style-prompt

| Флаг | Поведение |
|---|---|
| `--style-prompt "..."` | Строка из CLI |
| `--style-prompt-file path.txt` | Читать prompt из файла |
| `--no-style-prompt` | Отключить prompt (чистый TTS) |
| (ничего) | Дефолтный prompt из config.py |

### Native prompt vs prefix fallback

- **Native** (Gemini по умолчанию): `prompt` и `input` передаются раздельно в request body
- **Prefix** (старый fallback): prompt конкатенируется с текстом в поле `input`

Gemini 3.1 Flash TTS поддерживает inline audio tags: `[whispers]`, `[laughs]`, `[excited]` и другие.

**Дефолтный style prompt (только Gemini):**

```text
Голос технического подкаста: спокойный, вдумчивый, живой и уверенный.
Тёплый мужской тембр, средний темп, ясная артикуляция, без театральности.
```

## Style prompt (OpenAI TTS)

**Не используется.** Для OpenAI TTS моделей `--style-prompt` игнорируется — текст передаётся как есть.

## Запуск

```powershell
# Gemini TTS — женский тёплый голос
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Sulafat"

# Gemini TTS — мужской дикторский
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Charon"

# OpenAI GPT-4o Mini TTS через OpenRouter
voiceover generate `
  --provider openrouter-tts `
  --model "openai/gpt-4o-mini-tts-2025-12-15" `
  --voice "nova"
```

## Ключ

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

## Как работает

### Gemini TTS (native prompt)

```
POST https://openrouter.ai/api/v1/audio/speech
{
  "model": "google/gemini-3.1-flash-tts-preview",
  "input": "<текст чанка>",
  "prompt": "<style prompt>",
  "voice": "Puck",
  "response_format": "pcm"
}
```

Поле `prompt` передаётся отдельно от `input`. Если `--no-style-prompt` — поле `prompt` не отправляется.

Gemini через OpenRouter принимает только `response_format="pcm"`. Пайплайн конвертирует PCM в MP3 через FFmpeg.

### OpenAI TTS

```
POST https://openrouter.ai/api/v1/audio/speech
{
  "model": "openai/gpt-4o-mini-tts-2025-12-15",
  "input": "текст чанка",
  "voice": "ash",
  "response_format": "pcm"
}
```

Style prompt не добавляется. Пайплайн конвертирует PCM в MP3 через FFmpeg.

## Multi-speaker (Gemini dialogue)

**OpenRouter `/api/v1/audio/speech` поддерживает только ОДИН голос на запрос
(одно top-level поле `voice`).** Поле `multi_speaker_voice_config` НЕ
является частью этого endpoint и НЕ поддерживается — OpenRouter игнорирует
его и синтезирует весь input одним голосом. Подтверждено live-прослушиванием
2026-08-22 (см. `docs/reports/2026-08-21-gemini-dialogue-live-acceptance.md`):
весь диалог был озвучен одним женским голосом `Kore`.

Гибридный payload (top-level `voice` + `multi_speaker_voice_config`) больше
не используется и не должен документироваться как рабочий. Двухголосый
`gemini-dialogue` сейчас BROKEN и перерабатывается; НЕ заявляй, что он
работает. План: один запрос на реплику (turn) с одним документированным
top-level `voice`; запланировано в
[docs/plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md](plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md).
Два голоса недоступны, пока фикс не внедрён и человек не прослушал результат.

Нативный multi-speaker существует только в прямом Gemini API
(`generation_config.speech_config` с двумя `{speaker, voice}`), но не через
OpenRouter.

### Граница доказательств

- **Offline/mocked contract proof:** сериализация гибридного payload
  покрывалась mocked тестами, но это не доказывало, что провайдер применяет
  два голоса; live-прослуш показал, что провайдер игнорирует
  `multi_speaker_voice_config`.
- **Live provider acceptance:** аудируемость двух голосов ещё не принята
  (2026-08-22) — live-приёмка отмечена FAILED/BLOCKED.
- **Volatile facts:** доступность модели, цены и точная схема запроса могут
  меняться у провайдера; не переноси цены/доступность из исторических
  примеров в новые обещания.

## Цены

Цены с реальных smoke-прогонов (2 чанка, тестовый сценарий):

| Модель | Стоимость | Длина | Цена/мин |
|---|---:|---:|---|
| `google/gemini-3.1-flash-tts-preview` | $0.0135 | 26.7 сек | ~$0.030/мин |
| `openai/gpt-4o-mini-tts-2025-12-15` | $0.00022 | 32.3 сек | ~$0.00041/мин |

Точная стоимость: `GET /api/v1/generation?id=...` → `total_cost`.
