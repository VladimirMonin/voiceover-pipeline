# Voiceover Pipeline — Документация

## Для агентов

| Документ | Содержание |
|---|---|
| [Agent CLI Contract](agent-cli-contract.md) | Контракт для машинного использования: TTS, timing и generic local ASR, включая Qwen3-ASR, Nemotron, JSON-ответы, коды завершения (0/2/10/11/20/30/40/50), stdout/stderr и правила безопасности |
| [Remotion Workflow](remotion-workflow.md) | Как агент Remotion использует pipeline: от сценария до captions, manifest.json как entry-point, запрет оценки duration по словам |
| [Troubleshooting](troubleshooting.md) | Типовые ошибки: exit codes, recovery paths, зависимости |

## Для разработчиков

| Документ | Содержание |
|---|---|
| [Synthetic ASR evaluation corpus](asr-evaluation-corpus.md) | Privacy-safe локальный fixture corpus: manifest, SHA-256, pause/noise metadata, provenance и license boundary |
| [WVM Slice 5 local-reference benchmark](wvm-slice5-local-reference-benchmark.md) | Owner-approved local-only manifest для 50 WVM Slice 5 cases: явный corpus root, SHA-256 и NOASSERTION boundary |
| [ADR-001: Generic local ASR](adr/ADR-001-generic-local-asr.md) | Решение о capability-aware локальном ASR: registry, typed hints, dependency/device policy и граница timing/alignment |
| [Generic local ASR implementation plan](plans/2026-08-15-generic-local-asr-implementation-plan.md) | Последовательные offline-first срезы реализации, тестовые и benchmark-гейты |
| [audio.cpp hybrid migration plan](plans/2026-08-16-audio-cpp-hybrid-migration-plan.md) | Runtime-neutral план переезда Qwen ASR, Nemotron, Qwen TTS и OmniVoice на audio.cpp с будущим MLX-драйвером и Faster-Whisper рядом |
| [Native Windows Nemotron and OmniVoice plan](plans/2026-08-20-native-windows-nemotron-omnivoice-plan.md) | Серия native Windows-задач без Docker/WSL: portable runtime, package/build gates, Nemotron prompt плюс word timestamps, OmniVoice clone/design и live-приёмка; статус: in progress — offline-фундамент зафиксирован, native live-приёмка pending |
| [Agent-first Gemini dialogue release plan](plans/2026-08-21-agent-first-gemini-dialogue-0.6.0-release-plan.md) | План релиза 0.6.0: двухголосый gemini-dialogue, cast-safe resume, JSON-контракт, OmniVoice workflows, синхронизация skill и версий — **SUPERSEDED 2026-08-22** (OpenRouter применяет один голос) |
| [Two-voice dialogue fix plan](plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md) | Фикс 2026-08-22: turn-by-turn (один запрос на реплику, один документированный `voice`); live-приёмка FAILED/BLOCKED, 0.6.0 held |
| [Local audio runtime contract](audio-cpp-runtime.md) | Контракт `LocalAudioRuntime`, закреплённая версия audio.cpp, выбор рабочего маршрута, откат и проверяемые сведения о сборке |
| [audio.cpp Qwen ASR container recipe](audio-cpp-qwen-container-recipe.md) | Проверенный immutable CUDA image, read-only model mounts, JSON adapter и конфигурация Qwen word timestamps без inference claim |
| [audio.cpp feasibility report](research/2026-08-15-audio-cpp-feasibility.md) | Проверенная оценка полного перехода, optional backend и изолированного spike для Qwen/Nemotron без замены Faster-Whisper |
| [audio.cpp hybrid consolidation addendum](research/2026-08-16-audio-cpp-hybrid-consolidation.md) | Уточнённая цель: общий runtime для Qwen ASR, Nemotron, Qwen TTS и OmniVoice при сохранении Faster-Whisper и cloud providers |
| [Agent Development Workflow](../doc/agent-workflow.md) | Безопасный цикл работы агента: scope, dirty tree, Kanban, проверки и отдельные approvals для Git/release |
| [Artifacts & Analysis](artifacts-and-analysis.md) | JSON-схемы всех артефактов, обработка аудио (PCM→MP3, обрезка тишины, склейка), цены, сравнение моделей |
| [Whisper Timing](whisper-timing.md) | Whisper CPU small: модели, установка, команды, device/compute, word timestamps, SRT |
| [Polza Models](polza-openai-audio-models.md) | Polza AI + OpenAI GPT Audio: голоса, цены в RUB, ограничения, особенности |
| [Polza TTS Models](polza-tts-models.md) | Polza AI: OpenAI TTS через `/audio/speech`, ElevenLabs Turbo 2.5 и Multilingual v2 через `/media` |
| [OpenRouter TTS](openrouter-tts-models.md) | OpenRouter TTS: Google Gemini, OpenAI GPT-4o Mini TTS — голоса, style prompt, цены |
| [Qwen Local](qwen-local-tts.md) | Qwen3-TTS локально: preset-голоса, клонирование голоса, бесплатно (GPU) |
| [OmniVoice Local TTS](omnivoice-local-tts.md) | Явный offline встроенный female style condition через pinned Linux CUDA container; Q8_0, provenance и platform boundary |

## Быстрый старт

### Установленный пакет (опубликован на PyPI)

```powershell
pip install voiceover-pipeline
# или: pipx install voiceover-pipeline
# или: uvx voiceover-pipeline doctor  (без установки)

# Проверить окружение
voiceover doctor --json

# Проверить сценарий
voiceover validate --script "script.md" --json

# Сгенерировать озвучку + тайминги
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "script.md" `
  --run-id "prod" `
  --with-timings `
  --word-timestamps `
  --json `
  --overwrite
```

### Локальная разработка (клон репозитория)

```powershell
cd C:\PY\voiceover-pipeline
uv sync --group dev --extra timing-whisper
uv run voiceover doctor --json
uv run voiceover generate ... --with-timings --json --overwrite
```

## Образцы аудио

Первый чанк каждого облачного провайдера (OGG Vorbis 24 kHz mono):

| Файл | Модель | Цена минуты |
|---|---|---|
| [polza-gpt-audio-mini-chunk-01.ogg](polza-gpt-audio-mini-chunk-01.ogg) | GPT Audio Mini (Polza) | 0.004 ₽/мин (anomalous) |
| [polza-gpt-audio-chunk-01.ogg](polza-gpt-audio-chunk-01.ogg) | GPT Audio (Polza) | 7.00 ₽/мин |
| [polza-elevenlabs-turbo-2-5-chunk-01.ogg](polza-elevenlabs-turbo-2-5-chunk-01.ogg) | ElevenLabs Turbo 2.5 (Polza) | 3.51 ₽/мин |
| [polza-elevenlabs-multilingual-v2-chunk-01.ogg](polza-elevenlabs-multilingual-v2-chunk-01.ogg) | ElevenLabs Multilingual v2 (Polza) | 7.57 ₽/мин |
| [polza-openai-gpt-4o-mini-tts-chunk-01.ogg](polza-openai-gpt-4o-mini-tts-chunk-01.ogg) | GPT-4o Mini TTS (Polza) | 1.07 ₽/мин |
| [openrouter-gemini-tts-chunk-01.ogg](openrouter-gemini-tts-chunk-01.ogg) | Gemini TTS (OpenRouter) | $0.030/мин |
| [openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg](openrouter-openai-gpt-4o-mini-tts-chunk-01.ogg) | GPT-4o Mini TTS (OpenRouter) | $0.00041/мин |

## OpenCode Skill

Для агентов автоматизации: [skills/voiceover-pipeline/SKILL.md](skills/voiceover-pipeline/SKILL.md).

Скачать `.skill` архив из GitHub Release assets.
