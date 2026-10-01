# Версионный лог навыка

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: совместимость с приложением, история изменений, актуальность данных.

## Совместимость

| Поле | Значение |
|---|---|
| **Целевая версия приложения** | voiceover-pipeline 0.7.0 |
| **Skill revision** | 2026-10-01 (v0.7.0) |
| **Минимальная версия CLI** | 0.4.0 |
| **Максимальная проверенная** | 0.7.0 (offline wheel metadata/help; dependency-clean install `BLOCKED_CACHE`) |

## Что актуально в этой версии навыка

- Облачные TTS routes плюс локальные `qwen-local` и `omnivoice-local`
- Локальные ASR routes: `faster-whisper`, `qwen-local`, `nemotron-local`
- Общий `audio.cpp` runtime для локальных non-Whisper моделей с GPU lease,
  lifecycle, receipts и отдельными Linux/native-Windows launchers
- 7 исторически протестированных моделей; текущая доступность сверяется отдельно
- Актуальные голоса, модели и флаги — только через `voiceover list providers/voices --json` и `voiceover <cmd> --help`; подсчёты в этом файле не гарантируют текущий каталог
- `list voices --json` контракт: `voices` как плоский массив + `voice_categories` объект
- ElevenLabs через Polza: async `/api/v1/media` (submit → poll → download)
- Polza TTS OpenAI: JSON base64 через `/api/v1/audio/speech`
- OpenRouter Gemini TTS: exact `model`/verbatim `input`/`voice`/`response_format`
  payload без style/profile/vibe prefix; raw MP3/WAV/PCM validation
- Voiceover metadata: `format: voiceover` frontmatter для provider/model/voice/fallback/style_prompt, auto-detect in validate/generate
- OpenRouter Gemini dialogue: `--format gemini-dialogue`, frontmatter speaker map, inline audio tags, strict UTF-8 byte validation
- Gemini prompting guide: voice direction skeleton, safe audio tags, emotion recipes, voice selection, chunking limits
- Stability layer: `run_state.json`, `generation.log`, локальные retries (платный TTS — один submit на часть без автоматического fallback), `--resume`, paid-submit attempt marker и overwrite guard, `status`, `concat`, `--limit-chunks`, `--dry-run-cost`, `--json-events`
- Cloud transcription: 4 провайдера распознавания — `faster-whisper` (локальный), `openrouter-whisper` (текст без таймкодов), `groq-whisper` (сегменты + слова), `xai-stt` (слова + confidence)
- `list timing-providers` показывает 4 провайдера с `timestamps` полем
- OpenRouter Whisper: CLI блокирует `timings`/`--with-timings` с exit code 40 (API не возвращает таймкоды)
- Endpoint dispatch: `openai/*` → `/audio/speech`, `elevenlabs/*` → `/media`
- `speech-parts`/`--text`: строгий YAML и DB-first нативный маршрут; Gemini 3.8 Flash через Polza — только явный эмпирический opt-in, по умолчанию `BLOCKED_PROVIDER_CONTRACT`

## Цены

Этот файл **не** публикует актуальные цены и **не** является потолком
стоимости перед платным запросом. Исторические smoke-замеры устаревают; тариф,
доступность и верхняя цена подтверждаются только на момент согласованного
платного вызова. Смотри `voiceover help providers.polza` и `voiceover history
costs`.

## История изменений

| Дата | Изменения |
|---|---|
| 2026-10-01 | **v0.7.0:** явный эмпирический opt-in `--allow-experimental-gemini-speech-parts` допускает `polza-tts/google/gemini-3.8-flash-tts` для `--text`/`speech-parts`: один scalar `voice` на POST, эффективная инструкция в отдельном (не документированном для Gemini) поле `instructions`, наблюдённый WAV → MP3, приватный response body/receipt до разбора и запрет следования редиректам. По умолчанию маршрут и Flash-Lite остаются `BLOCKED_PROVIDER_CONTRACT`; флаг записывается в снимок, поэтому resume не может сменить политику. Не заявлять stable Gemini-маршрут или документацию инструкций. |
| 2026-10-01 | **S10 (unreleased):** упакованная атомарная справка `voiceover help [TOPIC] [--raw|--json]` читается из установленного пакета без ключей, `.env`, рабочего каталога и сети. README, docs index и skill приведены к фактическому разрешению секрета (непустое окружение процесса → явный `--env-file PATH` → `<CWD>/.env`, без поиска по родительским каталогам) и к явному согласию владельца на сетевые установки/загрузки моделей; из справочных файлов убраны волатильные цены и счётчики тестов. |
| 2026-09-29 | **Development/unreleased, source-only:** платный TTS записывает attempt marker перед submit, не делает автоматический повтор через retry или fallback voice; неподтверждённый исход блокирует resume/overwrite, локальные retries сохранены. Нет live/paid provider acceptance или заявления о релизе. |
| 2026-08-31 | **v0.6.1 candidate:** dialogue ASR quality gate удаляет из expected text только непроизносимые audio tags, нормализует `ё/е` и допускает эквивалентное деление ASR-токенов (`OmniVoice` / `Omni Voice`). Реальная лишняя речь и повторы остаются fail-closed. |
| 2026-08-24 | OpenRouter dialogue переведён на строгий verbatim turn input: style/profile/vibe/labels/соседний текст не попадают в synthesis request. Перед final concat обязателен явный per-turn ASR quality gate с transcript-free receipt; вставки, пропуски и повторы дают exit `60`. |
| 2026-08-24 | Исторический промежуточный transport fix перевёл Gemini 3.1 Flash TTS на `model`/`input`/`voice`/`response_format="pcm"` и raw audio response. Его prefix-style поведение в тот же день superseded строгим verbatim contract строкой выше; empty и wrapped JSON/data URI/SSE по-прежнему fail closed. |
| 2026-08-24 | Реализован offline contract канонического `dialogue`: `gemini-dialogue` сохранён alias, OpenRouter и OmniVoice выполняют turn-by-turn, final audio включает PCM паузы, resume использует fail-closed `synthesis_identity`, а receipts не публикуют private text. Владелец разрешил публикацию `0.6.0` до финального human audible PASS; post-publication live gate остаётся открытым и не подменяется transport/static доказательствами. |
| 2026-08-24 | OmniVoice Russian design теперь fail-closed отклоняет English accent/Chinese dialect conditioning и long-form requests выше 30 estimated seconds до provider/GPU; short design остаётся warning + experimental. JSON перечисляет clone/preset/short experimental/other-provider alternatives без неявной подмены. Добавлены path-free execution provenance и transcript-free `verify-tts`; hallucination не объявляется исправленной, technical PASS не заменяет human listening. |
| 2026-08-22 | **0.6.0 held:** live-приёмка двухголосого `gemini-dialogue` FAILED — OpenRouter игнорирует `multi_speaker_voice_config` и синтезирует одним голосом. Двухголосый результат недоступен до фикса по плану `docs/plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md`; не заявлять, что два голоса работают. |
| 2026-08-17 | Добавлен hybrid `audio.cpp` контур: Qwen3-ASR, Nemotron, Qwen3-TTS и OmniVoice; разделены offline/static/live evidence и Linux/Windows routes; внесены measured benchmark boundaries; для локального TTS цифры, версии, проценты и дроби теперь обязательно нормализуются в произносимые слова до генерации. |
| 2026-07-31 | Каноническая версия навыка закреплена в репозитории; справочник TTS и распознавания разделён для соблюдения лимита 300 строк; шаблон `.env.example` дополнен безопасными placeholder-ами Groq/xAI; runtime Hermes привязывается к in-repo skill. |
| 2026-05-27 | **v0.5.0:** Cloud transcription providers. `--timing-provider` (faster-whisper | openrouter-whisper | groq-whisper | xai-stt), `list timing-providers` с 4 провайдерами, архитектура `TranscriptionProvider` ABC. Groq Whisper — сегментные и пословные таймкоды через прямой API ($0.04/час). xAI STT — пословные таймкоды с confidence через xAI API. OpenRouter Whisper — блокировка `timings`/`--with-timings` (exit code 40) из-за отсутствия таймкодов. |
| 2026-05-10 | Добавлен Gemini prompting guide: `AUDIO PROFILE`/`SCENE`/`PERFORMANCE`/`CONTEXT`/`TRANSCRIPT`, safe audio tags, emotion recipes, voice selection, chunking guidance. |
| 2026-05-10 | Добавлен generic `format: voiceover` для single-speaker режимов: provider/model/voice в frontmatter, CLI overrides, full-error validator, backward compatibility с plain Markdown. |
| 2026-05-10 | Добавлен Gemini dialogue workflow: two speakers через OpenRouter `multi_speaker_voice_config` + обязательный top-level `voice`, full-error validator, `--speaker-voice`, `--agent`, chunk byte safety gates. **Устарело с 2026-08-22:** OpenRouter игнорирует `multi_speaker_voice_config` (один голос на запрос); двухголосый диалог BROKEN, перерабатывается по плану turn-by-turn (`docs/plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md`). |
| 2026-05-10 | Добавлен generation supervisor: state/log after each chunk, безопасный `--resume`, защита paid chunks от overwrite, `status`, `concat`, `--limit-chunks`, `--dry-run-cost`. ~~Universal retry для всех провайдеров~~ — **устарело**: оплаченная попытка резервируется до POST, неопределённый submit не повторяется автоматически; локальные retries отделены от paid. |
| 2026-05-09 | Gemini native prompt: отдельное поле `prompt` в request body вместо конкатенации в `input`. Флаги `--style-prompt-file`, `--no-style-prompt`. `prompt_mode` в manifest. Расширяемость под будущие Google/Polza модели. |
| 2026-05-01 | UV-first Python: `uv python install 3.12` вместо winget. Remotion scene grouping: Whisper-сегменты группируются по смысловым сценам, не по чанкам. Torch CPU-only диагностика. Qwen голоса обновлены до 9 актуальных. ~~Агент сам создаёт `.env` из `.env.example`~~ — **Устарело:** агент не создаёт, не копирует и не читает реальный `.env`; пользователь ведёт приватный env-файл, порядок — окружение процесса → явный `--env-file` → `<CWD>/.env` (см. `docs/03-security-and-secrets.md`). |
| 2026-04-29 | Добавлены `polza-tts`, ElevenLabs, OpenRouter OpenAI TTS. Обновлены все цены, голоса, workflows, evaluation. |
| 2026-03 | Исходная версия навыка под voiceover-pipeline 0.3.x (Polza GPT Audio, OpenRouter Gemini, Qwen) |
