# Voiceover Pipeline

Standalone CLI для генерации озвучки + Whisper timing из Markdown-сценариев.

Три TTS-провайдера: Polza GPT Audio, OpenRouter Gemini, Qwen3-TTS (GPU).
Whisper CPU small — точные тайминги для Remotion-анимаций и субтитров.
Agent-grade JSON-контракт: `--json`, semantic exit codes, `manifest.json`.

## Установка

```powershell
cd C:\PY\voiceover-pipeline
uv sync
uv sync --group timing-whisper          # Whisper timings
uv sync --group voiceover-qwen          # Qwen (GPU only)
```

Ключи API в `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-...
POLZA_API_KEY=pza_...
```

## Golden Command

```powershell
uv run voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "in\script.md" `
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

## Что на выходе

```
out/<run-id>/
├── manifest.json                          ← entry-point для агентов
├── <run-id>-voiceover-<model>.mp3         ← полный MP3
├── <run-id>-voiceover-<model>.json        ← run-манифест
├── <run-id>.timings.json                  ← Whisper-сегменты (ms)
├── <run-id>.srt                           ← субтитры SRT
└── chunks/
    ├── chunk_01.mp3 … chunk_NN.mp3
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
| Qwen3-TTS | CustomVoice (preset/clone) | Бесплатно (GPU) |
| Polza | `openai/gpt-audio-mini` | ~0.71 ₽ |
| Polza | `openai/gpt-audio` | ~7.63 ₽ |
| OpenRouter | `google/gemini-3.1-flash-tts-preview` | ~$0.03 |

## Тестирование

```powershell
uv sync --group dev --group timing-whisper
uv run --group dev pytest
```

45 тестов: JSON-контракт, exit codes, валидация, output policy.

## Документация

- [Agent CLI Contract](docs/agent-cli-contract.md) — JSON-контракт, exit codes, stdout/stderr, safety rules
- [Remotion Workflow](docs/remotion-workflow.md) — как агент Remotion использует pipeline
- [Artifacts & Analysis](docs/artifacts-and-analysis.md) — JSON-схемы, обработка аудио, сравнение моделей
- [Whisper Timing](docs/whisper-timing.md) — модель, device, compute, word timestamps
- [Troubleshooting](docs/troubleshooting.md) — типовые ошибки и их коды
- [Polza Models](docs/polza-openai-audio-models.md) — голоса, цены, особенности
- [OpenRouter Gemini](docs/openrouter-gemini-tts.md) — голоса, style prompt, цены
- [Qwen Local](docs/qwen-local-tts.md) — preset-голоса, клонирование, GPU
