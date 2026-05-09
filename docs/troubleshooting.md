# Troubleshooting

## FFmpeg / FFprobe

```
RuntimeError: FFmpeg is required but was not found in PATH.
```

Exit code: 11.

**Решение:**
1. Скачать с [ffmpeg.org](https://ffmpeg.org/download.html)
2. Добавить `ffmpeg.exe` и `ffprobe.exe` в `PATH`
3. Проверить: `ffmpeg -version`, `ffprobe -version`

## faster-whisper

```
ModuleNotFoundError: No module named 'faster_whisper'
```

Exit code: 10.

**Решение:**
```powershell
pip install voiceover-pipeline[timing-whisper]
```
Для локальной разработки: `uv sync --extra timing-whisper`

## Polza ключ

```
POLZA_API_KEY not found. Set it in .env: POLZA_API_KEY=...
```

Exit code: 20.

**Решение:** заполнить `.env`: `POLZA_API_KEY=pza_...`

`.env` ищется в CWD и вверх по родительским директориям.

## OpenRouter ключ

```
OPENROUTER_API_KEY is required
```

Exit code: 20.

**Решение:** заполнить `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

`.env` ищется в CWD и вверх по родительским директориям.

## CUDA не доступна

`doctor --json` показывает `"cuda": {"ok": false, "required": false}`.

Это **не блокирует** cloud TTS и CPU timings. Поле `workflow_ok` будет `true`.

CUDA нужна только для:
- `qwen-local` провайдера
- `--timing-device cuda`

**Решение для Qwen:** установить CUDA Toolkit и драйвер NVIDIA.
**Решение для Whisper:** использовать `--timing-device cpu` (дефолт).

## Первый запуск долгий

Whisper small (~486 MB) скачивается из HuggingFace при первом использовании. Последующие запуски используют кэш.

## PowerShell не показывает кириллицу

В `.srt` и `.timings.json` кириллица в UTF-8. PowerShell по умолчанию использует другую кодировку.

**Решение:**
```powershell
Get-Content "file.srt" -Encoding UTF8
```

Или открыть файл в любом редакторе.

## Папка уже существует

```
Run folder already exists: ... Use --overwrite to replace or --skip-existing.
```

Exit code: 30.

**Решение:** добавить `--overwrite` (удалить и создать заново) или `--skip-existing` (пропустить, вернуть JSON).

## Timing failure после успешной генерации

```
Voiceover generated but timing extraction failed: ...
```

Exit code: 40. MP3 уже сохранён.

**Восстановление:**
```powershell
voiceover timings --audio "out\<run>\*-voiceover-*.mp3"
```
При локальной разработке: `uv run voiceover timings --audio "out\<run>\*-voiceover-*.mp3"`

## Модель Whisper не скачана

При первом запуске `voiceover timings` модель загружается автоматически. При ошибках сети:

1. Проверить интернет
2. Попробовать ещё раз — используется retry с hf-mirror.com

## OpenRouter generation cost не приходит сразу

OpenRouter асинхронно обновляет usage. Пайплайн делает до 4 попыток с паузой 3 секунды. Если cost не получен, он будет `null` в JSON — это нормально, повторите проверку позже.

## --json stdout не парсится

При `--json` stdout должен содержать ровно один JSON object. Если появляются посторонние строки (прогресс-логи), обновите `voiceover-pipeline` до последней версии или используйте парсинг только последней строки как workaround.

## Invalid --run-id

```
{"status": "error", "error": "Invalid --run-id: ...", "code": 2}
```

Exit code: 2.

Разрешённые имена: `[a-zA-Z0-9._-]`. Запрещены: пробелы, точки в конце, спецсимволы, path separators, Windows reserved names (CON, AUX, COM1, etc). Подробности: [agent-cli-contract.md](agent-cli-contract.md#--run-id-rules).

## Invalid --output-dir

```
{"status": "error", "error": "--output-dir cannot be ...", "code": 2}
```

Exit code: 2.

Нельзя использовать drive root, home directory и CWD как `--output-dir`. Можно: `out`, `C:\PY\voiceover-pipeline\out`.

## Пустой CLI

```
usage: ...: error: the following arguments are required: command
```

Exit code: 2.

Нужно указать subcommand: `generate`, `split`, `timings`, `doctor`, `validate`, `list`.

## Gemini prompt body ошибка

```
Style prompt failed for chunk_01; retrying with shorter podcast style prompt.
```

Gemini может отклонить слишком длинный style prompt. Пайплайн автоматически ретраит с укороченным fallback-промптом. Если и это не помогает — попробуйте `--no-style-prompt`.

## Unsupported prompt mode

Если модель не поддерживает выбранный `prompt_mode`, пайплайн молча использует безопасный fallback:
- `openai/*` → `none` (prompt игнорируется)
- `google/*` → `native` (раздельный prompt + input)
- остальные → `none`
