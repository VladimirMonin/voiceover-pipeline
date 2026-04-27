# Changelog

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
