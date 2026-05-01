# Установка voiceover-pipeline и всех зависимостей

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: probe commands, install decision tree, OS specifics, выбор сборки.

## Правило установки

Агент САМ проверяет и устанавливает всё, что можно.
Пользователя просить выполнить команду ТОЛЬКО если среда не позволяет:
нет прав, интерактивный installer требует GUI, корпоративная политика запрещает.
Если агент может установить через winget/brew/apt/curl — ставит сам.

## Probe Commands

| Что | Windows | macOS/Linux |
|---|---|---|
| Python | `python --version` или `py -3 --version` | `python3 --version` |
| UV | `uv --version`, `uvx --version` | `uv --version`, `uvx --version` |
| FFmpeg | `ffmpeg -version`, `ffprobe -version` | `ffmpeg -version`, `ffprobe -version` |
| CLI | `voiceover doctor --json` | `voiceover doctor --json` |

## Installation Decision Tree

Проверяй последовательно, на каждом шаге запускай probe-команду:

**Шаг 1 — CLI уже работает?**
`voiceover doctor --json`
Если OK → пропустить установку, перейти к проверке extras.

**Шаг 2 — Попробовать uvx**
`uvx voiceover-pipeline doctor --json`
Если OK → использовать uvx для всех команд:
`uvx --from "voiceover-pipeline[timing-whisper]" voiceover-pipeline generate ...`

**Шаг 3 — Установить UV**
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
После установки открыть новый terminal, повторить шаг 2.

**Шаг 4 — Fallback: pipx**
Если UV не ставится или среда без UV:
`pipx install "voiceover-pipeline[timing-whisper]"`
Если `pipx` не найден: `python -m pip install pipx`, затем `pipx ensurepath`.
Если уже установлен base без extras: `pipx reinstall "voiceover-pipeline[timing-whisper]"` или `pipx uninstall voiceover-pipeline && pipx install "voiceover-pipeline[timing-whisper]"`.
После установки: `voiceover doctor --json`.

**Шаг 5 — Fallback: pip (последний)**
`python -m pip install --user "voiceover-pipeline[timing-whisper]"`
Или для project venv: `uv pip install "voiceover-pipeline[timing-whisper]"`.
Проверить: `voiceover doctor --json`.
Если `voiceover` не найден: `python -m pip show voiceover-pipeline`, затем новый terminal.

## Prerequisite Install (по ОС)

### Python ≥3.12

**UV-first (предпочтительно).** UV сам управляет версиями Python и скачивает недостающие:

- `uv python install 3.12` — установить Python 3.12 (управляемая установка)
- `uv venv --python 3.12` — создать venv, автоматически скачает 3.12 если нет
- После установки UV: `uv venv` может скачать последнюю версию даже при отсутствии Python в системе

**Fallback на системные менеджеры (если UV недоступен):**

- Windows: `winget install Python.Python.3.12`
  Если `python` не найден после установки: `py -3 --version`, открыть новый terminal.
- macOS: `brew install python@3.12`
- Linux: `apt install python3.12 python3-pip` (Debian/Ubuntu)
  Альтернативы: `dnf install python3.12` (Fedora), `pacman -S python` (Arch).

### FFmpeg + FFprobe

- Windows: `winget install ffmpeg`
  Если команда не найдена после установки: открыть новый terminal.
  Альтернатива: `winget install Gyan.FFmpeg` (полный build с ffprobe).
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg` (Debian/Ubuntu)
  Альтернативы: `dnf install ffmpeg` (Fedora), `pacman -S ffmpeg` (Arch).

### UV

- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
Если curl/irm заблокированы: `pipx install uv` или `pip install uv`.

## Выбор сборки voiceover-pipeline

| Extras | Добавляет | Для чего |
|---|---|---|
| (базовая) | CLI + все провайдеры + облачные TTS | Озвучка без таймингов |
| `timing-whisper` | faster-whisper | Whisper-тайминги из аудио |
| `voiceover-qwen` | Qwen3-TTS (PyTorch, transformers, soundfile) | Локальный бесплатный TTS на GPU |
| `cuda` | CUDA-зависимости для Qwen | GPU-ускорение для Qwen |

**Правило выбора:**

- Только облачная озвучка → Base (без extras)
- Нужны тайминги → + `timing-whisper`
- Нужен Qwen (GPU) → + `voiceover-qwen`
- Qwen + тайминги → + `timing-whisper`, `voiceover-qwen`
- Всё сразу → + `timing-whisper`, `voiceover-qwen`, `cuda`

Для production-видео: **всегда ставь `timing-whisper`**.

## Команды установки пакета

Console scripts: `voiceover` и `voiceover-pipeline` (работают оба).

| Менеджер | Base | +Whisper | +Qwen GPU | +All extras |
|---|---|---|---|---|
| **uvx** | `uvx voiceover-pipeline doctor` | `uvx --from "voiceover-pipeline[timing-whisper]" voiceover-pipeline generate --with-timings ...` | `uvx --from "voiceover-pipeline[voiceover-qwen]" voiceover-pipeline generate --provider qwen-local ...` | `uvx --from "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]" ...` |
| **pipx** | `pipx install voiceover-pipeline` | `pipx install "voiceover-pipeline[timing-whisper]"` | `pipx install "voiceover-pipeline[voiceover-qwen]"` | `pipx install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **pip** | `pip install voiceover-pipeline` | `pip install "voiceover-pipeline[timing-whisper]"` | `pip install "voiceover-pipeline[voiceover-qwen]"` | `pip install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **uv pip** | `uv pip install voiceover-pipeline` | `uv pip install "voiceover-pipeline[timing-whisper]"` | `uv pip install "voiceover-pipeline[voiceover-qwen]"` | `uv pip install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |

Рекомендация: используй `uvx` для разовых запусков, `pipx` для постоянной установки.

## Проверка установки

```powershell
# Базовая
voiceover doctor --json

# С таймингами
voiceover doctor --with-timings --json

# Для конкретного провайдера
voiceover doctor --provider polza-chat-audio --with-timings --json
voiceover doctor --provider polza-tts --with-timings --json
voiceover doctor --provider openrouter-tts --with-timings --json
voiceover doctor --provider qwen-local --json
```

Агент опирается на `workflow_ok`. Если `false` → `docs/09-troubleshooting.md`.

## Qwen-local отдельно

- Требует NVIDIA GPU + CUDA drivers (~4 GB VRAM).
- Агент НЕ чинит CUDA-драйверы молча.
- Проверить: `nvidia-smi`, `voiceover doctor --provider qwen-local --json`.
- Если CUDA unavailable → предложить cloud Polza/OpenRouter.
- Установка: `voiceover-pipeline[voiceover-qwen]`.
- Модель (~3.4 GB) скачивается из HuggingFace при первом запуске.

## First-run особенности

- Whisper модель (~486 MB, `small`) скачивается из HuggingFace при первом `--with-timings`.
- Qwen модель (~3.4 GB) скачивается при первом `--provider qwen-local`.
- Скачивание может занять несколько минут при первом запуске.
- Модели кешируются, повторные запуски быстрые.
- `.env` ищется в CWD и вверх по родительским директориям.
  Если `.env` нет — агент создаёт `.env.example` (см. `docs/03-security-and-secrets.md`).
