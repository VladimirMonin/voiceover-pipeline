# Changelog

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
- Added OpenRouter Gemini multi-speaker payload support with `multi_speaker_voice_config` while preserving OpenRouter's required top-level `voice` field.
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
