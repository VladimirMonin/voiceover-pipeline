# Remotion Workflow

Как агент Remotion может использовать `voiceover-pipeline` от сценария до готовых таймингов для анимаций.

## Setup для Remotion агента

```powershell
pip install "voiceover-pipeline[timing-whisper]"
```

После установки доступны консольные скрипты `voiceover` и `voiceover-pipeline`.

## Полный пайплайн

```text
Сценарий (Markdown, разделённый ******)
       │
       ▼
voiceover generate --with-timings
       │
       ├──► MP3 чанки + склеенный MP3
       ├──► chunks.json (позиции чанков)
       ├──► .timings.json (Whisper сегменты с точными ms)
       ├──► .srt (субтитры)
       └──► manifest.json (entry-point)
                │
                ▼
        Remotion agent
         читает timings.json
       │
       ├──► scene durations = segments[].duration_ms
       ├──► scene start/end = segments[].start_ms/end_ms
       ├──► captions = .srt
       └──► per-chunk timing = chunks.json chunks[].start_ms/end_ms
```

## Команда для агента

```powershell
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "C:\remotion-project\script.md" `
  --run-id "production" `
  --output-dir "out" `
  --with-timings `
  --timing-model small `
  --timing-device cpu `
  --json `
  --overwrite
```

## Что читать агенту

### 1. `manifest.json` — все пути

Пример (фактические пути зависят от `--output-dir` и `--run-id`):

```json
{
  "full_mp3": "C:\\voiceover-output\\production\\production-voiceover-openai-gpt-audio-mini.mp3",
  "timings_json": "C:\\voiceover-output\\production\\production.timings.json",
  "srt": "C:\\voiceover-output\\production\\production.srt",
  ...
}
```

### 2. `.timings.json` — источник истины для длительностей
```json
{
  "segments": [
    {"start_ms": 0, "end_ms": 4200, "duration_ms": 4200, "text": "..."},
    {"start_ms": 4200, "end_ms": 8300, "duration_ms": 4100, "text": "..."}
  ]
}
```

**Использовать:** `duration_ms` для длительности сцен, `start_ms`/`end_ms` для синхронизации.

### 3. `chunks.json` — соответствие чанков и позиций
```json
{
  "chunks": [
    {"id": "chunk_01", "start_ms": 0, "end_ms": 16920, "transcript": "..."}
  ]
}
```

**Использовать:** для сопоставления чанков сценария с реальными позициями в аудио.

### 4. `.srt` — субтитры

Импортировать напрямую в Remotion или использовать для captions.

## Тайминги vs оценка

| Источник | Точность | Используй для |
|---|---|---|
| `.timings.json` segments | Высокая (Whisper, ms) | **Production durations** |
| `chunks.json` start_ms | Высокая (ffprobe, ms) | Chunk alignment |
| "слова в секунду" | Низкая | **Только draft** |

**Правило:** если есть `.timings.json` — используй его. Не оценивай длительности по количеству слов.

## Пример: создание scene plan

```python
import json

# 1. Прочитать manifest.json — он знает все пути
manifest = json.load(open("production/manifest.json"))

# 2. Прочитать timings.json — точные millisecond-тайминги
timings = json.load(open(manifest["timings_json"]))

# 3. Создать scene plan
for seg in timings["segments"]:
    scene = {
        "start_ms": seg["start_ms"],
        "end_ms": seg["end_ms"],
        "duration_ms": seg["duration_ms"],
        "narration": seg["text"],
        "words": seg.get("words"),  # word-level highlights (опционально)
    }
```

## Provider выбор для Remotion

| Провайдер | Когда | Команда |
|---|---|---|
| `polza-chat-audio` (GPT Audio Mini) | Быстро и дёшево (0.71 RUB/min) | `voiceover generate --provider polza-chat-audio --model "openai/gpt-audio-mini"` |
| `openrouter-tts` (Gemini) | Западные голоса, качество ($0.03/min) | `voiceover generate --provider openrouter-tts --model "google/gemini-3.1-flash-tts-preview" --voice "Puck"` |
| `qwen-local` | Бесплатно, нужен GPU | `voiceover generate --provider qwen-local` |
