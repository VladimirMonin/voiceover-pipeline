# Qwen3-TTS локально

Qwen/Qwen3-TTS — open-source модель синтеза речи. Работает локально на GPU (NVIDIA, CUDA), полностью бесплатно.

Требуется extras: `pip install voiceover-pipeline[voiceover-qwen]`

## Модели

| Модель | HuggingFace ID | Размер | Назначение |
|---|---|---|---|
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | Готовые голоса — 24 пресета |
| Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | Клонирование голоса из референс-аудио |

Обе модели: 12Hz frame rate, поддержка 70+ языков (включая русский), до 30 секунд аудио за проход.

## Голоса (preset)

| Голос | Пол | Характер |
|---|---|---|
| `Aiden` | М | Спокойный, тёплый (**дефолт**) |
| `Alina` | Ж | Мягкий |
| `Amelia` | Ж | Чистый |
| `Arthur` | М | Глубокий |
| `Callum` | М | Энергичный |
| `Carter` | М | Уверенный |
| `Elijah` | М | Ровный |
| `Ethan` | М | Ясный |
| `Evelyn` | Ж | Тёплый |
| `Isabella` | Ж | Мелодичный |
| `Jack` | М | Живой |
| `James` | М | Солидный |
| `Landon` | М | Мягкий |
| `Liam` | М | Молодой |
| `Lily` | Ж | Нежный |
| `Lucas` | М | Яркий |
| `Mason` | М | Низкий |
| `Mia` | Ж | Лёгкий |
| `Natalia` | Ж | Выразительный |
| `Olivia` | Ж | Плавный |
| `Paul` | М | Чёткий |
| `Sofia` | Ж | Мягкий |
| `Theo` | М | Спокойный |
| `Violet` | Ж | Воздушный |

## Клонирование голоса (clone)

Для режима clone нужен референс-аудиофайл. Подойдёт 10-30 секунд чистой речи одного диктора.

```powershell
voiceover generate --provider qwen-local --mode clone --sample "my_voice_sample.mp3"
```

Если референсного текста нет (`--sample-text` пуст) — используется `x_vector_only_mode=True`, модель клонирует тембр без привязки к тексту референса.

```powershell
# С точным текстом референса (точнее клонирование):
voiceover generate --provider qwen-local --mode clone `
  --sample "my_voice_sample.mp3" `
  --sample-text "Текст, который произносится в референсе."
```

## Как работает

1. Загружается модель Qwen3-TTS (1.7B параметров) на GPU
2. Для каждого чанка вызывается `generate_custom_voice` или `generate_voice_clone`
3. Результат — WAV через `soundfile`
4. WAV конвертируется в MP3 через FFmpeg
5. Временные WAV-файлы сохраняются в `temp/`

## Требования к GPU

- NVIDIA GPU с CUDA support
- ~4 GB VRAM для CustomVoice (1.7B параметров в bfloat16)
- ~4 GB VRAM для Base (clone)
- Модель скачивается из HuggingFace один раз и кешируется

Без GPU модель упадёт на CPU — слишком медленно для практического использования.

## Цена

Бесплатно. В JSON-артефактах: `cost: 0.0, cost_currency: "RUB", cost_source: "qwen-local (free)"`.
