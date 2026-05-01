# Диагностика и решение проблем

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: doctor-guided recovery, command-not-found, dependency repair, provider/API, filesystem.

## Правило диагностики

Агент САМ чинит всё, что может.
Пользователя спрашивает ТОЛЬКО для: API-ключа, прав администратора,
GUI-инсталлятора, CUDA-драйверов.
Никогда не просит «выполните эту команду», если сам может выполнить.

## Doctor-guided Recovery

Запусти `voiceover doctor --provider <X> --with-timings --json`.
Смотри на `checks` в JSON-ответе:

| Check | Если `ok: false` + `required: true` | Действие |
|---|---|---|
| `python` | Python отсутствует | Установи Python ≥3.10 (см. `docs/02-install.md`) |
| `ffmpeg` | FFmpeg не найден | Установи FFmpeg + открой новый terminal |
| `ffprobe` | FFprobe не найден | Установи FFmpeg (идёт в комплекте) |
| `env_file` | `.env` отсутствует | Создай `.env.example`, попроси ключи ОДИН раз (см. `docs/03-security-and-secrets.md`) |
| `polza_key` | POLZA_API_KEY missing | Попроси добавить в `.env` ключ `pza_...`, НЕ читай `.env` |
| `openrouter_key` | OPENROUTER_API_KEY missing | Попроси добавить в `.env` ключ `sk-or-v1-...`, НЕ читай `.env` |
| `faster_whisper` | Whisper не установлен | Переустанови с extra `timing-whisper` |
| `cuda` | CUDA отсутствует | CUDA не нужна для cloud TTS; для Qwen — предложи Polza/OpenRouter |

Поля `required: false` + `ok: false` — только warning, не блокируют `workflow_ok`.

## Command Not Found

### `voiceover` / `voiceover-pipeline` не найдена

1. `python -m voiceover_pipeline.cli doctor --json` (diagnostic fallback)
2. `python -m pip show voiceover-pipeline` (проверить установлен ли)
3. Если не установлен: `pip install "voiceover-pipeline[timing-whisper]"`
4. Если установлен, но не найден: открыть новый terminal (PATH refresh)
5. Лучше использовать `uvx --from "voiceover-pipeline[timing-whisper]" voiceover-pipeline doctor --json`

### `uvx` / `uv` не найдена

1. Установить UV (см. `docs/02-install.md`)
2. Если PowerShell script blocked: попробовать `pipx install uv`
3. Если curl/irm blocked: `pip install uv` 
4. После установки: новый terminal

### `pipx` не найден

`python -m pip install pipx && pipx ensurepath`
Открыть новый terminal.

### `ffmpeg` / `ffprobe` не найден

1. Установить (см. `docs/02-install.md`)
2. `winget install ffmpeg` (Win), `brew install ffmpeg` (macOS), `apt install ffmpeg` (Linux)
3. После установки: открыть новый terminal
4. Проверить: `ffmpeg -version`, `ffprobe -version`

## Dependency Missing

### faster-whisper (code 10)

```
ModuleNotFoundError: No module named 'faster_whisper'
```

- Установить: `pip install "voiceover-pipeline[timing-whisper]"`
- Для uvx: `uvx --from "voiceover-pipeline[timing-whisper]" voiceover-pipeline ...`
- Для pipx: `pipx reinstall "voiceover-pipeline[timing-whisper]"` (если уже установлен без extras)
- Только при работе внутри репозитория voiceover-pipeline: `uv sync --extra timing-whisper`

### torch / CUDA (для Qwen)

- Проверить: `nvidia-smi`
- Проверить: `voiceover doctor --provider qwen-local --json`
- Установить: `voiceover-pipeline[voiceover-qwen,cuda]`
- ~4 GB VRAM требуется
- Агент НЕ чинит CUDA-драйверы молча — предложи cloud fallback

### torch installed as CPU-only (частый баг [cuda] extra)

**Симптом:** `torch.cuda.is_available()` → False, хотя nvidia-smi показывает GPU.
Причина: PyPI по умолчанию ставит CPU-сборку torch, даже с extras `cuda`.

**Диагностика:**
```powershell
python -c "import torch; print('version:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```
Если `torch.version.cuda is None` → torch CPU-only.

**Исправление:**
```powershell
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu128 --reinstall torch
```
После переустановки повторить: `python -c "import torch; print(torch.cuda.is_available())"`.

### soundfile / transformers (для Qwen)

- Установить: `pip install "voiceover-pipeline[voiceover-qwen]"`
- При ошибке: `pip install soundfile transformers`

## Model Downloads (first-run)

### Whisper model (~486 MB)

- Скачивается автоматически при первом `--with-timings`
- Источник: HuggingFace (hf-mirror.com как fallback)
- Если долго: это нормально, не зависание
- Если ошибка сети: проверить интернет, повторить
- Если диск полон: освободить место (~500 MB)

### Qwen model (~3.4 GB)

- Скачивается при первом `--provider qwen-local`
- Проверить диск: нужно ~4 GB свободно
- HuggingFace кеш: `~/.cache/huggingface/`

## Provider / API Errors

### Missing key (code 20)

```
POLZA_API_KEY not found / OPENROUTER_API_KEY is required
```

- НЕ читай `.env`
- Запусти `voiceover doctor --provider <X> --json`
- Если `polza_key.ok: false` — попроси добавить `POLZA_API_KEY=pza_...`
- Если `openrouter_key.ok: false` — попроси добавить `OPENROUTER_API_KEY=sk-or-v1-...`
- Больше не спрашивать

### Invalid key / 401 / 403

- Ключ есть, но неверный или нет доступа к модели
- Попроси пользователя проверить ключ в личном кабинете провайдера
- Polza: https://polza.ai/ → личный кабинет
- OpenRouter: https://openrouter.ai/keys

### Rate limit / provider down (code 30)

- Подождать и попробовать снова
- Сменить провайдера: Polza → OpenRouter, или наоборот
- Qwen-local не зависит от облачных лимитов

### OpenRouter cost `null` (не ошибка)

- Нормально: OpenRouter асинхронно обновляет usage
- Пайплайн делает до 4 попыток с паузой 3 секунды
- Если cost не получен — он `null` в JSON, `status` при этом `success`
- Повторить проверку позже, если нужна точная стоимость

### Style prompt rejected (OpenRouter)

```
No successful provider responses
```

- Автоматически пробуется укороченный fallback prompt
- Можно задать свой: `--style-prompt "..."` покороче и проще

## Output / Filesystem

### Папка уже существует (code 30)

- `--overwrite` — удалить и пересоздать (осторожно!)
- `--skip-existing` — пропустить, вернуть `status: skipped`
- Новый `--run-id` — создать рядом

### Permission denied (code 50)

- Проверить права на `--output-dir`
- Проверить свободное место на диске
- Не использовать `C:\`, home, CWD как output-dir

### Invalid --run-id (code 2)

- Только `[a-zA-Z0-9._-]`
- Без пробелов, path separators, Windows reserved names
- Примеры: `prod`, `prod-01`, `prod_01`, `prod.v1`

### Invalid --output-dir (code 2)

- Запрещено: drive root, home, CWD
- Разрешено: `out`, `out/project`, абсолютные пути вне CWD/home/root

## Whisper Timing Failure

### Exit 40 — Whisper упал, но MP3 сохранён

- **MP3 уже на диске!** Не запускай генерацию заново
- Восстановление: `voiceover timings --audio "out/<run-id>/<run-id>-voiceover-<model>.mp3" --run-id "prod" --json --overwrite`

### Exit 10 — faster-whisper отсутствует

- `pip install "voiceover-pipeline[timing-whisper]"`

## PowerShell / Terminal

### Кириллица не читается

- `.srt` и `.timings.json` в UTF-8, PowerShell по умолчанию не UTF-8
- `Get-Content "file.srt" -Encoding UTF8`
- Открыть в любом редакторе — файлы корректны

### PATH не обновлён после winget/brew

- Открыть **новый terminal** — PATH обновляется только при запуске оболочки
- Не использовать тот же терминал после установки

## Если ничего не помогает

1. `voiceover doctor --json` — полный diagnostic output
2. `python --version`, `ffmpeg -version`, `ffprobe -version`
3. `python -m pip show voiceover-pipeline`
4. Проверить `.env` через `doctor` (НЕ читая файл)
5. Проверить интернет (HuggingFace, Polza, OpenRouter могут быть недоступны)
