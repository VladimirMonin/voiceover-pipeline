# Remotion Agent Flow: от сценария до сцен

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Полный поток для интеграции voiceover-pipeline с Remotion.
> Универсальный для всех провайдеров (замени `<PROVIDER>` и `<MODEL>`).

## Шаг 1: Установка

```powershell
pip install "voiceover-pipeline[timing-whisper]"
```

## Шаг 2: Проверка окружения

```powershell
voiceover doctor --provider <PROVIDER> --with-timings --json
```

Убедись что `workflow_ok: true`.

## Шаг 3: Валидация сценария

```powershell
voiceover validate --script "script.md" --json
```

Если `valid: false` — покажи пользователю issues, не продолжай.

## Шаг 4: Генерация озвучки + таймингов

Polza Chat Audio (рубли, дёшево):

```powershell
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "script.md" `
  --run-id "production" `
  --output-dir "out" `
  --with-timings `
  --timing-model small `
  --timing-device cpu `
  --word-timestamps `
  --json `
  --overwrite
```

Polza TTS (рубли, классический TTS):

```powershell
voiceover generate `
  --provider polza-tts `
  --model "openai/gpt-4o-mini-tts" `
  --voice "ash" `
  --script "script.md" `
  --run-id "production" `
  --with-timings `
  --word-timestamps `
  --json `
  --overwrite
```

## Шаг 5: Чтение артефактов

```python
import json

# Entry-point — manifest.json знает все пути
manifest = json.load(open("out/production/manifest.json"))

# Точные тайминги в миллисекундах
timings = json.load(open(manifest["timings_json"]))

# Субтитры
srt_path = manifest["srt"]

# Чанки для per-scene alignment и цен
chunks = json.load(open(manifest["chunks_json"]))
```

## Шаг 6: Создание scene plan для Remotion

```python
scenes = []
for seg in timings["segments"]:
    scenes.append({
        "start_ms": seg["start_ms"],
        "end_ms": seg["end_ms"],
        "duration_ms": seg["duration_ms"],
        "narration": seg["text"],
        "words": seg.get("words")  # word-level highlights
    })
```

## Шаг 7: Использование в Remotion

- `scene["duration_ms"]` → `<Sequence durationInFrames={msToFrames(scene["duration_ms"])}>`
- `scene["words"]` → синхронизированная подсветка слов
- `srt_path` → `<Subtitles src={srt_path} />` или парсинг в кастомный компонент

## Важные правила

1. **НЕ оценивай длительность по словам.** Есть `.timings.json` → используй его.
2. **НЕ гадай имена файлов.** Читай `manifest.json` → он знает все пути.
3. **НЕ игнорируй exit codes.** Если `generate` упал с 40 — MP3 сохранён,
   запусти `voiceover timings --audio` отдельно.
4. **НЕ перезаписывай output без `--overwrite`.** Используй `--skip-existing`.
5. **Выбор провайдера** — читай `docs/05-providers-and-models.md` для 7 моделей и цен.
