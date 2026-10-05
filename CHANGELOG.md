# Changelog

## 0.8.0

- Google Gemini 3.8 Flash TTS (`google/gemini-3.8-flash-tts`) and Flash-Lite
  (`google/gemini-3.8-flash-lite-tts`) are ordinary speech models on both
  `polza-tts` and `openrouter-tts`: `voiceover list providers` lists them and
  `generate`/`validate` admit them without any experimental opt-in. Every request
  still carries exactly one scalar `voice` with the part's effective direction in
  a separate `instructions` field, and different parts or dialogue turns become
  separate requests, so no request ever implies more than one speaker. The former
  `--allow-experimental-gemini-speech-parts` flag is now accepted and recorded
  only as a deprecated compatibility spelling: it is never required and refuses
  nothing.
- `generate` and `validate` share one effective model admission rule: an unknown,
  stale, or provider-mismatched model is refused identically by both before any
  key read or request, so a model `generate` would reject is never reported as a
  usable route.
- Added optional path-only credential source defaults `VOICEOVER_POLZA_ENV_FILE`
  and `VOICEOVER_OPENROUTER_ENV_FILE`. Each holds a path the operator already
  maintains; when set it replaces the working-directory `.env` for that
  provider's secret without copying or merging keys. No new secret store and no
  parent-directory search were added.
- OpenRouter `/audio/speech` persists the bounded private raw response (the raw
  body, then its bounded receipt) before the status check and decode, keeps the
  bounded container the response reported so a local replay decodes the stored bytes
  as they arrived, maps the documented `audio/pcm` stream to canonical pcm16 24 kHz
  mono, and never follows a redirect, so a paid response is kept with one bounded
  refusal instead of a second POST.
- Includes the prior Windows portability and type repairs that keep the runtime
  and history checks portable on Windows, committed before this release.

## 0.7.0

- Added canonical SQLite-first generation and local history: reserve each paid
  attempt before submit, retain private response and receipt before parsing,
  recover saved results without repeating an unconfirmed paid POST, and report
  observed Decimal costs without inventing unknown amounts.
- Added strict `speech-parts` YAML, one-part `--text`, per-part voice selection,
  MP3/WAV finalization and DB-derived run artifacts. The optional Polza Gemini
  3.8 Flash path requires `--allow-experimental-gemini-speech-parts`: each part
  sends **one scalar voice per POST**, with spoken `input` separate from an
  `instructions` field that Polza does not document for Gemini. It is an
  empirical opt-in, not a stable provider guarantee or multi-speaker POST.
- Added local Qwen ASR provenance, offline FTS5 lexical search over saved speech
  and directions, and packaged `voiceover help`. Semantic/hybrid/vector search
  remains deferred to a later release.
- Offline end-to-end tests cover FFmpeg, SQLite, search, exact costs and safe
  recovery; a six-minute experimental three-voice MP3 received a positive
  overall listening assessment. That recording used a one-use runner, not the
  newly packaged CLI. External provider billing, undocumented instruction
  behavior, other operating systems and an offline dependency-clean install
  remain separate, unverified boundaries.

## 0.6.1

- Fixed false dialogue quality failures for non-spoken audio tags, Russian
  `ё`/`е` ASR drift, and equivalent ASR token boundaries such as
  `OmniVoice` / `Omni Voice`.
- Extra speech, omissions and repeated phrases still fail closed.
- No new TTS request is needed to re-check already generated trusted turns;
  `--resume` reruns the per-turn ASR gate before concat.

## 0.6.0

> **0.6.0 publication authorized by the owner on 2026-08-24.** The package is
> published before the final audible dialogue check. OpenRouter dialogue now
> uses one paid request and one `voice` per turn, sends only the exact current
> turn text, and requires a per-turn ASR quality gate before concat. This
> release does not claim that two voices have passed final human listening;
> the post-publication live check remains open while OpenRouter returns 502.

### OmniVoice Local Voice Bank

- Added `--voice-provider omnivoice-local` with four local modes: `auto` (default preset), bank preset (named OmniVoice voices), `clone` (voice from a reference audio), and `design` (build a synthetic voice from properties).
- OmniVoice is single-speaker per run; local multi-speaker dialogue is not first-class in this release.

### Native Windows Acceptance

- Native Windows launcher paths for local audio.cpp models are accepted as first-class (Qwen3-TTS, OmniVoice, Qwen3-ASR, Nemotron), alongside the existing Linux routes.

### Hardened Gemini Dialogue Workflow

- Dialogue validation now enforces exactly two distinct speakers and rejects duplicate speaker voices.
- Style prompt is capped with a strict byte limit to catch oversized prompts before paid generation.
- Top-level Gemini `voice` is derived from the speaker map with conflict rejection instead of trusting a manual override.
- Hardened `--json`/`--json-events` handling: JSON error envelope for failures and rejection of `--json` combined with `--json-events`.
- Canonical dialogue resume identity (`--resume`) with fail-closed behavior (exit code 30) when the run cannot be safely resumed.

### Agent Skill Reconciliation

- Skill updated: Two-speaker podcast branch, OmniVoice workflows (auto/bank/clone/design), and safe timings guidance.

### Tests

- Added mocked end-to-end Gemini dialogue generation test (no live provider calls).

## 0.5.1

### Documentation & Skill Update

- Updated skill documentation: 4 timing providers (faster-whisper, openrouter-whisper, groq-whisper, xai-stt) in providers-and-models, commands-and-flags, security-and-secrets, install, version-log.
- Updated `list timing-providers` JSON contract with 4 providers and `timestamps` field.
- Added `--timing-provider` choices for `groq-whisper` and `xai-stt` in all CLI reference tables.
- Documented openrouter-whisper blocking (exit code 40) — no timestamps returned by API.
- Created `.env.example` with `GROQ_API_KEY` and `X_AI_API_KEY` placeholders.
- Updated AGENTS.md / OpenCode skill in project and Obsidian vault.

### New Providers

- **groq-whisper**: Direct Groq API with segment + word timestamps, `whisper-large-v3-turbo` at $0.04/hr.
- **xai-stt**: xAI STT API with word-level timestamps + confidence, 12 audio formats.

## 0.5.0

### Cloud Transcription Providers

- Added `--timing-provider openrouter-whisper` for cloud transcription via OpenRouter Whisper Large V3 Turbo.
- Added `list timing-providers` command with local + cloud provider display.
- Implemented `TranscriptionProvider` ABC architecture for provider extensibility.
- OpenRouter Whisper: JSON body with base64-encoded audio.

## 0.4.5

### Gemini Prompting Guide

- Added a practical Gemini 3.1 Flash TTS prompting guide for voice direction, scene/profile/performance structure, English audio tags, emotion recipes, voice selection, chunk sizing, and production anti-patterns.
- Added README guidance for Gemini prompting: treat Gemini TTS as a directed voice performer, keep service directions separate from transcript, use `PERFORMANCE` for global emotion and inline tags for local delivery changes.
- Expanded default Gemini dialogue safe tags with practical production tags such as `[thoughtfully]`, `[medium pause]`, `[long pause]`, `[gasp]`, `[cough]`, `[uhm]`, `[panicked]`, `[trembling]`, and `[shouting]`.
- Updated bundled and installed OpenCode skills to point agents at the new Gemini prompting guide.

### Tests

- 120 pytest tests, including coverage for the expanded Gemini dialogue safe tag set.

## 0.4.4

### Generation Stability

- Added `run_state.json` with atomic writes after every successfully saved chunk, including chunk number, file, duration, generation id, model, voice, text, and text hash.
- Added `generation.log` in every run folder; it is written even when `--json` is enabled.
- Added universal per-chunk retry wrapper for all providers with `--retries`, `--retry-delay`, `--retry-max-delay`, and `--no-retry`.
- Added `--resume` to continue interrupted runs without regenerating completed chunks; resume rejects changed scripts when `run_state.json` exists.
- Added paid-audio overwrite protection: `--overwrite` refuses to delete existing chunks unless `--confirm-delete-paid-audio` is also set.
- Added `voiceover status --run-id ...` and `voiceover concat --run-id ... --format ogg` for partial run inspection and safe partial audio assembly.
- Added `--dry-run-cost`, `--limit-chunks`, and `--json-events` for safer agent workflows and long-running generation visibility.
- Fixed Whisper install guidance from `uv sync --group timing-whisper` to `uv sync --extra timing-whisper`.

### Tests

- 117 pytest tests covering metadata validation, Gemini dialogue validation, backward compatibility for plain Markdown, provider payload regressions, retry/resume safety, state persistence, logging, status, dry-run limits, and partial concat.

## 0.4.3

### Unified Script Metadata

- Added `format: voiceover` frontmatter for single-speaker providers: provider/service, model, voice, fallback voice, style prompt, and chunk limits can now live in the Markdown script.
- Added auto-detection for metadata scripts in `validate` and `generate`; plain delimiter-based Markdown remains backward compatible.
- Added full-error validation for metadata scripts so agents receive all provider/model/voice/chunk defects in one JSON report.
- Added CLI overrides for metadata scripts: `--provider`, `--model`, and `--voice` override frontmatter.

### Gemini Dialogue

- Added `format: gemini-dialogue` for OpenRouter Gemini 3.1 Flash TTS with two speakers, per-speaker Gemini voices, shared style prompt, and inline emotion tags.
- Added OpenRouter Gemini multi-speaker payload support with `multi_speaker_voice_config` while preserving OpenRouter's required top-level `voice` field. *(Note 2026-08-22: this payload never worked — OpenRouter ignores `multi_speaker_voice_config` and applies a single voice; the 0.6.0 two-voice release is held.)*
- Added strict UTF-8 byte validation for Gemini dialogue chunks to catch oversized final chunks before paid generation.

### Tests

- 109 pytest tests covering metadata validation, Gemini dialogue validation, backward compatibility for plain Markdown, and provider payload regressions.

## 0.4.2

### Gemini Native Prompt Support

- `TTS_PROMPT_MODE_NONE / PREFIX / NATIVE` — три режима prompt для TTS провайдеров
- `PROMPTABLE_TTS_MODELS` карта в `config.py`: Gemini Flash TTS по умолчанию использует `native` prompt (отдельное поле `prompt` в request body)
- `OpenRouterTTSProvider` теперь принимает `prompt_mode` и строит тело запроса через `build_request_body()` вместо конкатенации строк
- Новый модуль `tts_prompting.py`: `resolve_prompt_mode()`, `build_request_body()`, `build_prompted_input()`, `read_style_prompt_from_file()`
- Старый fallback `prefix` сохранён для обратной совместимости

### CLI: новые флаги для style-prompt

- `--style-prompt-file path/to/prompt.txt` — читать prompt из файла (удобно для длинных WVM-промптов)
- `--no-style-prompt` — отключить prompt полностью (для чистого TTS в тестах)
- Приоритет: `--no-style-prompt` > `--style-prompt-file` > `--style-prompt` > дефолт из `config.py`

### Расширяемость

- `POLZA_PROMPTABLE_TTS_MODELS` — заготовка для будущих promptable моделей Polza
- `resolve_prompt_mode()` по префиксу модели: `google/*` → `native`, `openai/*` → `none`
- Неизвестные Google-модели (например `google/gemini-2.5-pro-tts`) автоматически используют `native` prompt

### Manifest

- `chunks.json` теперь содержит поле `prompt_mode` в метаданных манифеста

### Tests

- 98 pytest tests (добавлены тесты на native/prefix/none режимы, `build_request_body`, CLI-флаги, unknown-модели)

## 0.4.1

### Skill Fixes

- UV-first Python: `uv python install 3.12` вместо winget, агент сам создаёт `.env`
- Remotion semantic scene grouping: Whisper-сегменты группируются по смысловым сценам, не по чанкам
- Torch CPU-only диагностика в troubleshooting
- Qwen голоса обновлены до 9 актуальных (из HuggingFace model card)
- Регрессионный набор расширен до 9 кейсов (R5-R9 из эксплуатации)

## 0.4.0

### New Providers & Models

- `polza-tts` — новый провайдер для классического text-to-speech через Polza AI:
  - `openai/gpt-4o-mini-tts` через `/api/v1/audio/speech` — JSON base64, ~1.07 ₽/мин
  - `elevenlabs/text-to-speech-turbo-2-5` через `/api/v1/media` — async task, URL download, ~3.51 ₽/мин
  - `elevenlabs/text-to-speech-multilingual-v2` через `/api/v1/media` — async task, URL download, ~7.57 ₽/мин
- `openrouter-tts` расширен моделью `openai/gpt-4o-mini-tts-2025-12-15` (~$0.00041/мин)
- Полный список голосов Gemini TTS (30 голосов)
- Полный список голосов ElevenLabs через Polza (21 имя, display-names из allowlist)

### Architecture

- `PolzaTTSProvider`: model-aware dispatch — `openai/*` → `/audio/speech`, `elevenlabs/*` → `/media`
- `OpenRouterTTSProvider`: model-aware voice defaults — `Puck` для Gemini, `alloy` для OpenAI TTS
- Style prompt пропускается для OpenAI TTS моделей в OpenRouter
- `/api/v1/media` для ElevenLabs: submit → poll → download, стоимость из `usage.cost_rub`

### JSON Contract

- `list voices` теперь возвращает `voices` как плоский массив (backward-compatible) + `voice_categories` как опциональный объект с разбивкой по семействам голосов

### Docs

- `docs/polza-tts-models.md` — полная документация по Polza TTS моделям
- `docs/openrouter-tts-models.md` — Gemini + OpenAI TTS через OpenRouter, с голосовыми таблицами
- Обновлены `docs/artifacts-and-analysis.md`, `docs/remotion-workflow.md`, `README.md`, `docs/README.md`
- Семплы OGG для всех 7 моделей (Vorbis 24kHz mono)
- Удалён устаревший `docs/openrouter-gemini-tts.md` (заменён)

### Tests

- 61 pytest tests (добавлены тесты на PolzaTTSProvider, `/api/v1/media` flow, voice defaults, contract)

### Package

- Версия: 0.4.0
- `/out` убран из sdist include
- Provider-specific model defaults (каждый провайдер знает свою модель по умолчанию)
- Model/provider validation — невалидная комбинация падает до API call
- Polza TTS: direct cost из `usage_direct` сохраняется в ChunkArtifact (не ждём history)
- Skill docs в репозитории (`docs/skills/voiceover-pipeline`), 15 файлов
- `.skill` архив (ZIP) для агентов OpenCode
- 67 pytest tests

## 0.3.0

### Agent-Grade CLI

- `--json` output: stdout содержит ровно один JSON object, stderr содержит progress/logs
- Semantic exit codes: `0` (success), `2` (invalid args), `10` (missing dep), `11` (no ffmpeg), `20` (no key), `30` (provider/run error), `40` (whisper error), `50` (output error)
- Non-JSON mode: human-readable errors в stderr
- `manifest.json` — entry-point со всеми путями к артефактам

### Whisper Timing

- Добавлен `voiceover timings --audio` для извлечения таймингов из готового MP3
- `--with-timings` в `generate` — генерация + тайминги одним заходом
- `.timings.json`: `segment_count`, `segments[].start_ms/end_ms/duration_ms/text`
- `.srt`: стандартный SubRip для Remotion и видеоредакторов
- `--word-timestamps`: word-level highlights для караоке-субтитров
- Default: `--timing-model small`, `--timing-device cpu`, `--timing-compute int8`
- Backend: `faster-whisper`

### Safe Output Policies

- `--overwrite` удаляет существующий run folder и создаёт заново
- `--skip-existing` возвращает `status: skipped` без изменения файлов
- Default: ошибка code 30 если папка существует
- `--run-id` validation: запрет абсолютных путей, `.`, `..`, separators, whitespace, trailing dot, Windows reserved names (CON/NUL/COM1..LPT9)
- `--output-dir` validation: запрет drive root, home, CWD
- `_safe_remove_run_dir`: guard для CWD, drive root, home, выход за output-dir

### Doctor Improvements

- `required_ok` / `optional_ok` / `workflow_ok` вместо `all_ok`
- CUDA optional по умолчанию, required только для `qwen-local` и `--timing-device cuda`
- `--provider`, `--with-timings`, `--timing-device` для workflow-aware проверки

### Tests

- 45 pytest tests: JSON contract, exit codes, validation, output policy
- Dev dependency: `pytest`
- Конфигурация: `uv sync --extra dev` / `pip install -e ".[dev]"`

### Documentation

- `README.md`: чистый entry-point с golden command
- `docs/agent-cli-contract.md`: строгий reference: команды, JSON, exit codes, stdout/stderr, safety rules, golden workflow
- `docs/troubleshooting.md`: exit codes к каждой ошибке, recovery paths
- `docs/remotion-workflow.md`: практическое руководство для Remotion агента
- `docs/artifacts-and-analysis.md`: связи артефактов, JSON-схемы, аудио-обработка
- Убраны stale refs: `video-001`, `opencode.json`, `all_ok`, `"segments": 8`

### Remotion Integration

- `manifest.json` как entry-point для агентов
- `.timings.json` как source of truth для scene durations
- `.srt` для captions
- Запрет оценки duration по words-per-second при наличии timings
- `voiceover-pipeline` добавлен в Remotion skill boundary
