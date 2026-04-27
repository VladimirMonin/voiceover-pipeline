# Whisper Timing для Remotion

Whisper CPU small (`faster-whisper`) — минимальный transcriber для получения точных таймингов из готового MP3. Точность текста вторична, тайминги первичны.

## Модели

| Модель | Параметры | Размер | CPU время* | Качество RU |
|---|---|---|---|---|
| `base` | 74M | ~148 MB | ~5× realtime | Экспериментальное (WER ~73%) |
| `small` | 244M | ~486 MB | ~2× realtime | Минимальное рабочее (**дефолт**) |
| `medium` | 769M | ~1.5 GB | ~1× realtime | Хорошее |
| `large-v3-turbo` | 809M | ~1.6 GB | ~0.7× realtime | Лучшее за скорость |
| `large-v3` | 1550M | ~3.1 GB | ~0.5× realtime | Максимальное |

*на CPU с int8; CUDA быстрее в 3-7×

## Установка

```powershell
pip install voiceover-pipeline[timing-whisper]
```

Для локальной разработки: `uv sync --extra timing-whisper`

Модель (~486 MB для `small`) скачивается из HuggingFace при первом запуске и кешируется.

## Команды

### Тайминги из готового MP3

```powershell
# CPU small (дефолт)
voiceover timings --audio "out\run\audio.mp3"

# Другая модель
voiceover timings --audio "out\run\audio.mp3" --model large-v3-turbo

# С параметрами
voiceover timings `
  --audio "out\run\audio.mp3" `
  --model small `
  --device cpu `
  --compute int8 `
  --language ru `
  --output-dir "out" `
  --run-id "мой-прогон"
```

### Генерация + тайминги одним заходом

```powershell
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "in\script.md" `
  --run-id "prod" `
  --with-timings `
  --timing-model small `
  --timing-device cpu `
  --word-timestamps
```

### Word-level timestamps

С флагом `--word-timestamps` каждый сегмент в `.timings.json` получает массив `words`:

```json
{
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 4200,
      "text": "Максимальное качество видео",
      "words": [
        {"word": "Максимальное", "start_ms": 0, "end_ms": 860},
        {"word": "качество", "start_ms": 860, "end_ms": 1480},
        {"word": "видео", "start_ms": 1480, "end_ms": 1860}
      ]
    }
  ]
}
```

Использовать для: word-level подсветки в Remotion, караоке-субтитров, точных акцентов.

## Device и compute

| `--timing-device` | Когда |
|---|---|
| `cpu` | Всегда работает, медленнее (**дефолт**) |
| `auto` | Авто-выбор: CUDA если есть GPU, иначе CPU |
| `cuda` | Только NVIDIA GPU |

| `--timing-compute` | Когда |
|---|---|
| `auto` | Авто детекция архитектуры GPU (**дефолт для `auto` device**) |
| `int8` | CPU и старые GPU, минимальный VRAM (**дефолт для CPU**) |
| `int8_float16` | Turing+ GPU (RTX 20xx+), 73% меньше VRAM |
| `float16` | Blackwell GPU (RTX 50xx), чистый FP16 |
| `float32` | Fallback при ошибках загрузки |

Алгоритм авто-детекции повторяет `Whisper-Voice-Machine`:
- CPU → `int8`
- CUDA Blackwell (CC ≥ 12.0) → `float16`
- CUDA pre-Turing (CC < 7.0) → `int8`
- CUDA Turing+ (CC 7.0-11.x) → `int8_float16`

При ошибке загрузки — fallback: `float32` (CPU) или `float16` (CUDA).

## Артефакты

### `.timings.json`

```json
{
  "artifact_type": "voiceover-timings",
  "model": "small",
  "backend": "faster-whisper",
  "device": "cpu",
  "compute_type": "int8",
  "total_duration_ms": 25520,
  "segment_count": 8,
  "segments": [
    {
      "id": 0,
      "start_ms": 0,
      "end_ms": 4200,
      "duration_ms": 4200,
      "text": "..."
    }
  ]
}
```

### `.srt`

Стандартный SubRip формат для импорта в видеоредакторы и Remotion.

```srt
1
00:00:00,000 --> 00:00:04,200
Текст первого сегмента

2
00:00:04,200 --> 00:00:08,300
Текст второго сегмента
```

## Использование в Remotion

Тайминги из `.timings.json`:
- `start_ms` / `end_ms` каждой сцены для анимаций
- `duration_ms` для длительности сцен
- `.srt` для captions/subtitles

Агент Remotion должен использовать эти точные значения вместо примерных оценок "слов в секунду".
