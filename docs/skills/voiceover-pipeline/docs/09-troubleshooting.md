# Диагностика и решение проблем

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: doctor-guided recovery, command-not-found, dependency repair, provider/API, filesystem.

## Правило диагностики

Агент сам чинит локальные неисправности в пределах разрешённого scope.
Права администратора, GUI-инсталлятор, CUDA-драйверы, сетевая установка,
загрузка модели, provider GET и платный POST требуют отдельного решения владельца.
Никогда не просит ключ в чат и не читает приватный env-файл. Для платного
вызова нужен подтверждённый тариф и безопасный потолок расходов; неопределённый
submit агент не повторяет автоматически.

## Doctor-guided Recovery

Только в одобренном окружении запусти
`voiceover doctor --provider <X> --with-timings --json`: приложение может
прочитать env-файл ради наличия ключа (агент его не читает). Без такого согласия
используй безопасную справку `voiceover help start.quick --json` и сообщи,
что проверка ключа `NOT_RUN`. Смотри на `checks` в JSON-ответе:

| Check | Если `ok: false` + `required: true` | Действие |
|---|---|---|
| `python` | Python отсутствует | Требуется Python ≥3.11 (см. `docs/02-install.md`) |
| `ffmpeg` | FFmpeg не найден | Установи FFmpeg + открой новый terminal |
| `ffprobe` | FFprobe не найден | Установи FFmpeg (идёт в комплекте) |
| `env_file` | файла env нет или выбранный источник непригоден (явный `--env-file` или настроенный `VOICEOVER_POLZA_ENV_FILE`/`VOICEOVER_OPENROUTER_ENV_FILE`) | Создай только `.env.example`; реальный env заводит владелец вне инструментов агента, агент его не читает. При непригодном явном `--env-file` doctor даёт exit `0`, `status: success`, но `workflow_ok: false` |
| `polza_key` | POLZA_API_KEY missing | Попроси владельца поместить `pza_...` в приватный env-файл/окружение; НЕ создавай и НЕ читай `.env` |
| `openrouter_key` | OPENROUTER_API_KEY missing | Попроси владельца поместить `sk-or-v1-...` в приватный env-файл/окружение; НЕ создавай и НЕ читай `.env` |
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

## Локальные модели: подготовить заранее

- Нативная `generate --with-timings` проверяет локальную модель до платного TTS:
  без заранее подготовленного кеша отказывает, а не скачивает веса после оплаты.
  Qwen-local тоже требует заранее подготовленные веса.
- Отдельная `timings --timing-provider faster-whisper` может загрузить веса
  при первом запуске. Это сеть: перед командой нужно отдельное разрешение
  владельца; без него — `NOT_RUN`, а не пробный вызов.
- Кеш HuggingFace обычно находится в `~/.cache/huggingface/`; отсутствие модели
  не доказывает неисправность API. Нельзя читать приватные модели или скачивать
  новые веса без согласия владельца.

## Provider / API Errors

### Missing key (code 20)

```
POLZA_API_KEY not found / OPENROUTER_API_KEY is required
```

- НЕ читай и не создавай `.env`
- Запусти `voiceover doctor --provider <X> --json`
- Если `polza_key.ok: false` — попроси владельца добавить `POLZA_API_KEY=pza_...` в приватный env/окружение процесса (либо настроить path-only `VOICEOVER_POLZA_ENV_FILE`)
- Если `openrouter_key.ok: false` — попроси владельца добавить `OPENROUTER_API_KEY=sk-or-v1-...` в приватный env/окружение процесса (либо настроить path-only `VOICEOVER_OPENROUTER_ENV_FILE`)
- Больше не спрашивать

### Invalid key / 401 / 403

- Ключ есть, но неверный или нет доступа к модели
- Попроси пользователя проверить ключ в личном кабинете провайдера
- Polza: https://polza.ai/ → личный кабинет
- OpenRouter: https://openrouter.ai/keys

### Rate limit / provider down (code 30)

- Платный POST с неизвестным исходом CLI НЕ повторяет автоматически: попытка
  остаётся `submitting`, а `--resume` блокируется (`PAID_SUBMIT_UNCONFIRMED`).
- Новую попытку делает только владелец явным решением, после проверки возможной
  прежней оплаты и доказанного нового потолка, с ДРУГИМ `--run-id` (папка с
  `pending_attempt` защищена).
- Смена провайдера — только новый явный прогон с разрешения владельца.
- Qwen-local/OmniVoice не зависят от облачных лимитов; их локальный retry отделён.

### OpenRouter cost `null` (не ошибка)

- Нормально: OpenRouter асинхронно обновляет usage
- Пайплайн делает до 4 попыток с паузой 3 секунды
- Если cost не получен — он `null` в JSON, `status` при этом `success`
- Дополнительный price GET допускается только как отдельно разрешённая сетевая
  операция; если он недоступен, стоимость остаётся `null`, а не оценкой.

### OpenRouter provider error

```
No successful provider responses
```

- OpenRouter `/audio/speech` не принимает style prompt:
  `--style-prompt`/`--style-prompt-file` отклоняются до платного запроса, а
  `--no-style-prompt` — no-op. Подача задаётся внутри сценария.
- Один turn делает ровно один платный synthesis POST; автоматического retry/fallback нет.
- Оборванный ответ не повторяй: `--resume` только при проверенном raw или известном
  Media ID, иначе новый `--run-id` с разрешения владельца.

## Output / Filesystem

### Папка уже существует (code 30)

- `--overwrite` — удалить и пересоздать (осторожно!); для платного прогона — только
  по явному решению владельца о потере и цене
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

### Сбой таймингов после TTS — проверяй код и сохранённое аудио

- Legacy timing error может дать exit `40`; нативный partial/result-preserved
  маршрут — exit `50`. Сначала проверь JSON/state/history и наличие MP3, не
  запускай TTS заново по одному числовому коду.
- Для локального восстановления используй `history resume` только если прогон
  и попытка допущены к resume; команда потенциально платна на других частях.
  Альтернатива: отдельный `timings --audio <saved-mp3> --run-id <new-id> --json`
  без `--overwrite` по папке платного прогона, с разрешением на возможную
  загрузку модели или облачный вызов.

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

1. В одобренном окружении `voiceover doctor --json` — diagnostic output (приложение может прочитать env-файл)
2. `python --version`, `ffmpeg -version`, `ffprobe -version`
3. `python -m pip show voiceover-pipeline`
4. Проверить наличие ключа через `voiceover doctor --json` (путь, не значение; файл не читать)
5. Проверить интернет (HuggingFace, Polza, OpenRouter могут быть недоступны)
