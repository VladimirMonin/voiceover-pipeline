# Voiceover Pipeline — Документация

## Для агентов

| Документ | Содержание |
|---|---|
| [Agent CLI Contract](agent-cli-contract.md) | Контракт для машинного использования: команды, JSON-ответы, exit codes (0/2/10/11/20/30/40/50), stdout/stderr, safety rules |
| [Remotion Workflow](remotion-workflow.md) | Как агент Remotion использует pipeline: от сценария до captions, manifest.json как entry-point, запрет оценки duration по словам |
| [Troubleshooting](troubleshooting.md) | Типовые ошибки: exit codes, recovery paths, зависимости |

## Для разработчиков

| Документ | Содержание |
|---|---|
| [Artifacts & Analysis](artifacts-and-analysis.md) | JSON-схемы всех артефактов, обработка аудио (PCM→MP3, обрезка тишины, склейка), цены, сравнение моделей |
| [Whisper Timing](whisper-timing.md) | Whisper CPU small: модели, установка, команды, device/compute, word timestamps, SRT |
| [Polza Models](polza-openai-audio-models.md) | Polza AI + OpenAI GPT Audio: голоса, цены в RUB, ограничения, особенности |
| [OpenRouter Gemini](openrouter-gemini-tts.md) | OpenRouter + Google Gemini TTS: голоса, style prompt, цены в USD, конвертация в MP3 |
| [Qwen Local](qwen-local-tts.md) | Qwen3-TTS локально: preset-голоса, клонирование голоса, бесплатно (GPU) |

## Быстрый старт

### Установленный пакет (опубликован на PyPI)

```powershell
pip install voiceover-pipeline
# или: pipx install voiceover-pipeline
# или: uvx voiceover-pipeline doctor  (без установки)

# Проверить окружение
voiceover doctor --json

# Проверить сценарий
voiceover validate --script "script.md" --json

# Сгенерировать озвучку + тайминги
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

### Локальная разработка (клон репозитория)

```powershell
cd C:\PY\voiceover-pipeline
uv sync --extra dev --extra timing-whisper
uv run voiceover doctor --json
uv run voiceover generate ... --with-timings --json --overwrite
```

## Образцы аудио

Первый чанк каждого облачного провайдера (OGG Vorbis 42 kbps mono):

| Файл | Модель | Цена минуты |
|---|---|---|
| [polza-gpt-audio-mini-chunk-01.ogg](polza-gpt-audio-mini-chunk-01.ogg) | GPT Audio Mini (Polza) | 0.71 ₽ |
| [polza-gpt-audio-chunk-01.ogg](polza-gpt-audio-chunk-01.ogg) | GPT Audio (Polza) | 7.63 ₽ |
| [openrouter-gemini-tts-chunk-01.ogg](openrouter-gemini-tts-chunk-01.ogg) | Gemini TTS (OpenRouter) | $0.03 |
