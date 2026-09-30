# Voiceover Pipeline — Документация

## Для агентов

| Документ | Содержание |
|---|---|
| [Agent CLI Contract](agent-cli-contract.md) | Машинный контракт TTS, timing, local ASR и offline `history` CLI: JSON/exit (0/2/10/11/20/30/40/50/60), stdout/stderr и границы безопасности |
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
| [Two-voice dialogue fix plan](plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md) | Turn-by-turn: один запрос, один `voice` и точный verbatim input на реплику; per-turn ASR gate до concat. Публикация 0.6.0 разрешена владельцем до повторной live-приёмки; human audible PASS не заявлен. |
| [VoiceOver Pipeline development plan](plans/2026-09-29-voiceover-pipeline-development-plan.md) | План последовательной доработки поверх `c64fd14`: команды озвучки и распознавания, история и расходы в SQLite, поиск по словам, документация, логирование, этапы с критериями закрытия и матрица проверок; целевой релиз `v0.7.0`. Статус: **IN_PROGRESS** — S00, S02, S03 и S04 приняты офлайн; S01 проверен только по source, его live/listening остаётся blocked; S05 принят родителем офлайн после устранения Sol6 P1; S06 закрыт офлайн-ядром, но Polza Gemini-маршрут `BLOCKED_PROVIDER_CONTRACT`; S07 Qwen ASR и provenance приняты **только офлайн**, реальные модели/Polza Qwen не проверены; S08 (полнотекстовый поиск FTS5) реализован и проверен офлайн; S09 (semantic/hybrid) вынесен из `v0.7.0` (`DEFERRED`) в отдельный план следующего релиза; live/listening остаются NOT_RUN. Фактические результаты — в stage-отчётах ниже. |
| [Semantic search — next-release plan](plans/2026-10-01-semantic-search-next-release-plan.md) | Отложенный этап S09 плана разработки как самостоятельный план следующего релиза: два backend (локальный `Qwen/Qwen3-Embedding-0.6B` и Polza `qwen/qwen3-embedding-8b`), явные query/document preprocessing, версионированные immutable-профили и кеш в SQLite, необязательный безопасный `sqlite-vec` `vec0`, приватность без неявной загрузки корпуса, инкрементальная идемпотентная индексация с учётом стоимости, фильтры до top-k и RRF-hybrid, офлайн fault-injection и approval-gated реальная приёмка. Статус: **plan-only**, `NOT_STARTED`, планируемое обозначение `v0.8.0` — только planning label, не approval. |
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
| [OmniVoice hallucination research](reports/2026-08-24-omnivoice-hallucination-research.md) | Upstream и exact-run evidence: unsupported Russian accent conditioning, long-form nonclaims и bounded quality gate |
| [S00 baseline и проверки](reports/2026-09-29-s00-baseline.md) | Исходное состояние поверх `3355c23`; после настройки среды — результаты pytest/Ruff/mypy и версия `0.6.1`; отдельно — статическая карта событий `generation.log` (не live-проверка). |
| [S01 source-only отчёт по внешним контрактам](reports/2026-09-29-s01-source-only.md) | Source-only проверка узких контрактов Gemini/Polza, Qwen ASR и двух embedding-режимов: что подтверждено кодом на `1ba5743`, что `BLOCKED_PROVIDER_CONTRACT`/`NOT_RUN`; live/listening не выполнялись. |
| [S02 офлайн-приёмка оплаченных запросов](reports/2026-09-29-s02-paid-safety.md) | На `141a4d5`: точные суммы, запрет повторного paid POST, GET-only Media и сохранённый raw до FFmpeg; тесты и известные `NOT_RUN`/пробелы отдельно. |
| [S03 офлайн-приёмка границ CLI и сервисов](reports/2026-09-29-s03-service-boundaries.md) | На `f373d54`: подготовка, единый paid TTS loop, ASR, recovery, цены и финализация вынесены в сервисы; CLI/JSON и paid-safety сохранены по офлайн-тестам, live/listening **NOT_RUN**. |
| [S04 офлайн-приёмка SQLite-истории и импорта](reports/2026-09-29-s04-local-history.md) | На `025ff62` проверены migration/Decimal/FK/WAL, приватный home, zero-write legacy preview, идемпотентный import и metadata-only history CLI; текущая генерация остаётся на JSON до S05, live/listening **NOT_RUN**. |
| [S05 DB-first исполнение и восстановление](reports/2026-09-30-s05-db-first-execution.md) | На `01bca32`: каноническая SQLite-история для всех фактически работающих свежих маршрутов TTS/ASR/timing/verify, единый владелец и paid-boundary, матрица диспетчеризации и ручной fake interruption→resume; live/listening/installed-model **NOT_RUN**, статус `ACCEPT_OFFLINE` после проверенного P1 fix (Sol6 BLOCK относился к прежним байтам, повторный review не выполнялся). |
| [S06 speech-parts, короткая реплика и audio-format](reports/2026-09-30-s06-speech-parts.md) | Офлайн-ядро (`speech-parts`, `--text`, pre-POST бюджет, DB-first resume и WAV) принято родителем; **полный S06 не принят**: Polza Gemini-маршрут остаётся `BLOCKED_PROVIDER_CONTRACT`, live/listening **NOT_RUN**. |
| [S07 Qwen ASR и происхождение таймкодов](reports/2026-10-01-s07-qwen-asr.md) | 0.6B/1.7B без подмены, long-form без выдуманных речевых spans, SQLite/ms и source-bound FTS; исправлены Sol6 P1, повторный review — `ACCEPT_OFFLINE`. Реальные локальные модели/Polza Qwen и listening **NOT_RUN**. |
| [S08 полнотекстовый поиск FTS5](reports/2026-09-30-s08-fts5-search.md) | На локальном `06817f0`: migration v3 `search_chunks`/`search_fts`/состояние, `search` (lexical, role, filters до limit, `ё→е`) и `index status/build/rebuild`; канонический текст не теряется при сбое производного индекса. Независимый Sol6 дал `ACCEPT_OFFLINE`, два P2 затем исправлены fail-first (повторный review `NOT_RUN`); `--mode semantic|hybrid` явно отложен (S09), live/listening **NOT_RUN**. |
| [S09 deferral: план следующего релиза](reports/2026-10-01-s09-deferred-plan.md) | Только документированный перенос semantic/hybrid/embeddings/`sqlite-vec` в отдельный план следующего релиза; Sol6 `ACCEPT_PLAN_ONLY`, реализации и live-проверок нет, v0.7.0 сохраняет обязательный S08 FTS5. |

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
