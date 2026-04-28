# Voiceover Pipeline

Standalone CLI для генерации озвучки + Whisper timing из Markdown-сценариев.

Четыре TTS-провайдера: Polza GPT Audio, Polza TTS, OpenRouter TTS, Qwen3-TTS (GPU).
Whisper CPU small — точные тайминги для Remotion-анимаций и субтитров.
Agent-grade JSON-контракт: `--json`, semantic exit codes, `manifest.json`.

## Install

Console scripts: `voiceover` and `voiceover-pipeline` (both work).

| Manager | Base | + Whisper | + Qwen GPU | + All extras |
|---|---|---|---|---|
| **uvx** (no install) | `uvx voiceover-pipeline doctor` | `uvx --from "voiceover-pipeline[timing-whisper]" voiceover-pipeline generate --with-timings ...` | `uvx --from "voiceover-pipeline[voiceover-qwen]" voiceover-pipeline generate --provider qwen-local ...` | `uvx --from "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]" ...` |
| **pipx** | `pipx install voiceover-pipeline` | `pipx install "voiceover-pipeline[timing-whisper]"` | `pipx install "voiceover-pipeline[voiceover-qwen]"` | `pipx install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **pip** | `pip install voiceover-pipeline` | `pip install "voiceover-pipeline[timing-whisper]"` | `pip install "voiceover-pipeline[voiceover-qwen]"` | `pip install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |
| **uv pip** | `uv pip install voiceover-pipeline` | `uv pip install "voiceover-pipeline[timing-whisper]"` | `uv pip install "voiceover-pipeline[voiceover-qwen]"` | `uv pip install "voiceover-pipeline[timing-whisper,voiceover-qwen,cuda]"` |

### From source

```powershell
git clone https://github.com/VladimirMonin/voiceover-pipeline
cd voiceover-pipeline
uv sync --group dev --extra timing-whisper
uv run voiceover doctor
```

### First run

- `.env` is searched in CWD and upwards through parent directories.
- Whisper model (~486 MB) auto-downloads from HuggingFace on first `--with-timings`. Subsequent runs use cache.
- Qwen-local requires NVIDIA GPU + CUDA drivers (~4 GB VRAM).

## API Keys

Create `.env` in your working directory (CWD or any parent directory — searched upwards):

```env
POLZA_API_KEY=pza_...
OPENROUTER_API_KEY=sk-or-v1-...
```

Never commit `.env`.

## Golden Command

```powershell
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

Результат (JSON stdout):

```json
{"status": "success", "provider": "polza-chat-audio", "run_id": "prod",
 "duration_ms": 25520, "segment_count": 8, "cost": {"total": 0.0146, "currency": "RUB"}}
```

Результат (JSON stdout):

```json
{"status": "success", "provider": "polza-chat-audio", "run_id": "prod",
 "duration_ms": 25520, "segment_count": 8, "cost": {"total": 0.0146, "currency": "RUB"}}
```

## Что на выходе

```
out/<run-id>/
├── manifest.json                          ← entry-point для агентов
├── <run-id>-voiceover-<model>.mp3         ← полный MP3
├── <run-id>-voiceover-<model>.json        ← run-манифест
├── <run-id>.timings.json                  ← Whisper-сегменты (ms)
├── <run-id>.srt                           ← субтитры SRT
└── chunks/
    ├── chunk_01.mp3 ... chunk_NN.mp3
    └── chunks.json
```

## Команды

| Команда | Зачем |
|---|---|
| `doctor` | Проверить окружение |
| `validate --script X` | Проверить сценарий |
| `list providers/voices/timing-models` | Доступные модели |
| `split --script X` | Чанки сценария |
| `generate` | Полная генерация + тайминги |
| `timings --audio X` | Тайминги из готового MP3 |

Все команды поддерживают `--json`.

## Модели и цены

| Провайдер | Модель | Цена минуты |
|---|---|---|
| Polza | `openai/gpt-audio-mini` | ~0.004 ₽/мин (anomalous 200s smoke — model added speech, needs rerun) |
| Polza | `openai/gpt-audio` | ~7.00 ₽/мин |
| Polza | `openai/gpt-4o-mini-tts` | ~1.07 ₽/мин |
| Polza | `elevenlabs/text-to-speech-turbo-2-5` | ~3.51 ₽/мин |
| Polza | `elevenlabs/text-to-speech-multilingual-v2` | ~7.57 ₽/мин |
| OpenRouter | `google/gemini-3.1-flash-tts-preview` | ~$0.030/мин |
| OpenRouter | `openai/gpt-4o-mini-tts-2025-12-15` | ~$0.00041/мин |
| Qwen3-TTS | CustomVoice (preset/clone) | Бесплатно (GPU) |

## Тестирование

```powershell
uv sync --group dev --extra timing-whisper
uv run pytest
```

61 тест: JSON-контракт, exit codes, валидация, output policy, providers.

## Agent Skill

OpenCode agent skill — устанавливает pipeline, выбирает провайдера, генерирует озвучку:

- Source: https://github.com/VladimirMonin/voiceover-pipeline/tree/v0.4.0/docs/skills/voiceover-pipeline
- Download: https://github.com/VladimirMonin/voiceover-pipeline/releases/tag/v0.4.0

## Legal / Provider Notes

This project is an independent CLI wrapper around third-party providers.
It is not affiliated with OpenAI, Google, OpenRouter, Polza.ai, Qwen, or CTranslate2 / faster-whisper.
Provider names and model names are used solely for integration and documentation purposes.
Generated audio usage is subject to the selected provider's terms of service.
Do not upload private voice samples or generated speech without permission.

## License

MIT. See [LICENSE](LICENSE).

## Документация

- [Agent CLI Contract](docs/agent-cli-contract.md) — JSON-контракт, exit codes, stdout/stderr, safety rules
- [Remotion Workflow](docs/remotion-workflow.md) — как агент Remotion использует pipeline
- [Artifacts & Analysis](docs/artifacts-and-analysis.md) — JSON-схемы, обработка аудио, сравнение моделей
- [Whisper Timing](docs/whisper-timing.md) — модель, device, compute, word timestamps
- [Troubleshooting](docs/troubleshooting.md) — типовые ошибки и их коды
- [Polza Models](docs/polza-openai-audio-models.md) — голоса, цены, особенности
- [Polza TTS Models](docs/polza-tts-models.md) — OpenAI TTS, ElevenLabs через Polza AI
- [OpenRouter TTS](docs/openrouter-tts-models.md) — Gemini, OpenAI TTS через OpenRouter
- [OpenCode Skill](docs/skills/voiceover-pipeline/SKILL.md) — Agent skill для установки и озвучки
- [Qwen Local](docs/qwen-local-tts.md) — preset-голоса, клонирование, GPU
