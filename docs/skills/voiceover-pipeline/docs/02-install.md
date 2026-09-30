# Установка voiceover-pipeline и всех зависимостей

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: probe commands, install decision tree, OS specifics, выбор сборки.

## Правило установки

Агент САМ проверяет окружение и предлагает установку. Любую сетевую установку
(`uv`/`uvx`, `curl`-installer, `winget`/`brew`/`apt`, `pipx`/`pip`) и загрузку
моделей выполняй **только с явного разрешения владельца** и в согласованном
окружении: без разрешения пропусти шаг и спроси. Пользователя просить выполнить
команду ТОЛЬКО если среда не позволяет: нет прав, интерактивный installer требует
GUI, корпоративная политика запрещает.

## Probe Commands

`voiceover doctor` может прочитать env-файл для проверки ключа: запускай только
в одобренном окружении (см. `03-security-and-secrets.md`). Если такого согласия
нет, проверяй наличие установленного CLI через `voiceover help --json` без
секретов и сети.

| Что | Windows | macOS/Linux |
|---|---|---|
| Python | `python --version` или `py -3 --version` | `python3 --version` |
| UV | `uv --version`, `uvx --version` | `uv --version`, `uvx --version` |
| FFmpeg | `ffmpeg -version`, `ffprobe -version` | `ffmpeg -version`, `ffprobe -version` |
| CLI | `voiceover doctor --json` | `voiceover doctor --json` |

## Installation Decision Tree

Проверяй последовательно, на каждом шаге запускай probe-команду:

**Шаг 1 — CLI уже работает?**
Сначала `voiceover help --json` (без env-файла). `voiceover doctor --json` —
только после согласия на проверку env в выбранном окружении. Если CLI есть,
пропустить установку и перейти к проверке extras.

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
| (базовая) | CLI + облачные TTS | Озвучка без таймингов |
| `timing-whisper` | `faster-whisper`, `ctranslate2` | Whisper-тайминги из аудио |
| `voiceover-qwen` | Qwen3-TTS (`torch`, `qwen-tts`, `soundfile`, `numpy`) | Локальный TTS на GPU |
| `asr-qwen` | `qwen-asr` | Локальное распознавание Qwen ASR |
| `asr-nemotron` | `accelerate`, `librosa`, `torch`, `transformers` | Локальное распознавание Nemotron |
| `cuda` | CUDA-библиотеки (Windows/Linux) | GPU-ускорение для локальных моделей |

**Правило выбора:**

- Только облачная озвучка → Base (без extras)
- Нужны тайминги → + `timing-whisper`
- Нужен локальный Qwen TTS → + `voiceover-qwen` (+ `cuda`)
- Нужно локальное распознавание → + `asr-qwen` ИЛИ `asr-nemotron`

`asr-qwen`, `voiceover-qwen` и `asr-nemotron` **взаимоисключающие** (см.
`[tool.uv] conflicts` в `pyproject.toml`): единого «all extras» набора нет,
ставьте только нужный локальный маршрут.

## Команды установки пакета

Console scripts: `voiceover` и `voiceover-pipeline` (работают оба).

| Менеджер | Base | +Whisper | +Qwen GPU | +Whisper+Qwen+cuda |
|---|---|---|---|---|
| **uv tool** | `uv tool install voiceover-pipeline` | `uv tool install "voiceover-pipeline[timing-whisper]"` | `uv tool install "voiceover-pipeline[voiceover-qwen]"` | `uv tool install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **uvx** | `uvx voiceover-pipeline doctor` | `uvx --from "voiceover-pipeline[timing-whisper]" voiceover-pipeline generate --with-timings ...` | `uvx --from "voiceover-pipeline[voiceover-qwen]" voiceover-pipeline generate --provider qwen-local ...` | `uvx --from "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]" ...` |
| **pipx** | `pipx install voiceover-pipeline` | `pipx install "voiceover-pipeline[timing-whisper]"` | `pipx install "voiceover-pipeline[voiceover-qwen]"` | `pipx install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **pip** | `pip install voiceover-pipeline` | `pip install "voiceover-pipeline[timing-whisper]"` | `pip install "voiceover-pipeline[voiceover-qwen]"` | `pip install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **uv pip** | `uv pip install voiceover-pipeline` | `uv pip install "voiceover-pipeline[timing-whisper]"` | `uv pip install "voiceover-pipeline[voiceover-qwen]"` | `uv pip install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |

Рекомендация: используй `uvx` для разовых запусков и `uv tool install` для постоянной изолированной установки. `pipx` — только fallback, если `uv tool` недоступен.

## Проверка установки

```powershell
# Базовая
voiceover doctor --json

# С таймингами (все timing-провайдеры)
voiceover doctor --with-timings --json
voiceover doctor --with-timings --timing-provider groq-whisper --json
voiceover doctor --with-timings --timing-provider xai-stt --json

# Для конкретного TTS-провайдера
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
- Модель (~3.4 GB) нужно заранее скачать/закешировать: неявной загрузки нет, и
  агент не докачивает её без разрешения владельца.

## First-run особенности

- Локальная модель Qwen (~3.4 GB) и интегрированные локальные тайминги
  (`generate --with-timings`, Whisper `small` ~486 MB) требуют заранее
  закешированных весов: маршрут проверяет кэш и **не** докачивает неявно, чтобы
  не заплатить за TTS до выяснения.
- Отдельный `timings --timing-provider faster-whisper` может скачать модель Whisper
  из HuggingFace при первом запуске: это сетевая операция, выполняемая с разрешения
  владельца.
- Модели кешируются, повторные запуски быстрые.
- Значение ключа разрешается так: непустое окружение процесса → явный
  `--env-file PATH` → `<CWD>/.env`; поиска по родительским каталогам нет. Реальный
  env-файл создаёт пользователь; агент создаёт только `.env.example`
  (см. `docs/03-security-and-secrets.md`).
