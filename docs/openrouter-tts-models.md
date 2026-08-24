# OpenRouter TTS

Текущий поддерживаемый text-to-speech route через OpenRouter: Google Gemini TTS.

## Модели

| Поле | Значение |
|---|---|
| ID | `google/gemini-3.1-flash-tts-preview` |
| Endpoint | `/api/v1/audio/speech` |
| Контекст | 32 000 токенов |
| Языки | 70+, включая русский |
| Исторический smoke | ~$0.030/мин (2026-04-29, не текущая котировка) |

`openai/gpt-4o-mini-tts-2025-12-15` больше не публикуется текущим
`/api/v1/models?output_modalities=speech` и отклоняется до запроса. Модели
`openai/gpt-audio-mini` и `openai/gpt-audio` принадлежат chat-audio контракту
`/chat/completions`; они не являются drop-in заменой для `/audio/speech`.

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

Клиент автоматически выбирает `Puck` для Gemini.

Важно:
- OpenAI voices вроде `alloy`, `ash`, `nova` не работают для Gemini.
- ElevenLabs voices вроде `Rachel`, `Aria` не работают для Gemini.
- Использовать только Gemini prebuilt voice names, в TitleCase.

> **Образцы:**
> - [openrouter-gemini-tts-chunk-01.ogg](openrouter-gemini-tts-chunk-01.ogg) — Gemini, голос `Puck`, ~$0.030/мин
> - [openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg](openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg) — исторический GPT-4o Mini TTS smoke; модель больше не предлагается текущим speech-каталогом

## Verbatim input (Gemini)

Для актуального OpenRouter `/audio/speech` поле `input` равно только точному
произносимому тексту. Style prompt, profile, vibe, speaker labels и соседние
реплики не отправляются ни префиксом, ни отдельным полем. Подача выбирается
top-level `voice`; явные `--style-prompt`/`--style-prompt-file` отклоняются.

Gemini 3.1 Flash TTS поддерживает inline audio tags: `[whispers]`, `[laughs]`, `[excited]` и другие.

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

```

## Ключ

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

## Как работает

### Gemini TTS

```
POST https://openrouter.ai/api/v1/audio/speech
{
  "model": "google/gemini-3.1-flash-tts-preview",
  "input": "<style prompt>\n\n<текст чанка>",
  "voice": "Puck",
  "response_format": "pcm"
}
```

Ответ — raw audio body, не JSON, data URI, base64 field или SSE. Пайплайн
явно запрашивает документированный `pcm`, валидирует raw MP3/PCM MIME-типы
и конвертирует PCM в MP3 через FFmpeg. Пустой или обёрнутый JSON/text/SSE
payload отклоняется без попытки угадать формат.

## Multi-speaker (Gemini dialogue)

**OpenRouter `/api/v1/audio/speech` поддерживает только ОДИН голос на запрос
(одно top-level поле `voice`).** Поле `multi_speaker_voice_config` НЕ
является частью этого endpoint и НЕ поддерживается — OpenRouter игнорирует
его и синтезирует весь input одним голосом. Подтверждено live-прослушиванием
2026-08-22 (см. `docs/reports/2026-08-21-gemini-dialogue-live-acceptance.md`):
весь диалог был озвучен одним женским голосом `Kore`.

Гибридный payload (top-level `voice` + `multi_speaker_voice_config`) больше
не используется и не должен документироваться как рабочий. Канонический
`dialogue` выполняется как один request на реплику (turn) с одним
документированным top-level `voice`; `gemini-dialogue` — compatibility alias.
Offline tests проверяют маршрутизацию, но audible PASS всё ещё требует human
listening acceptance и не должен подменяться metadata или mocked response.

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
| `openai/gpt-4o-mini-tts-2025-12-15` (исторический, withdrawn) | $0.00022 | 32.3 сек | ~$0.00041/мин |

Точная стоимость: `GET /api/v1/generation?id=...` → `total_cost`.
