# Remotion Agent Flow: от сценария до сцен

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Полный поток для интеграции voiceover-pipeline с Remotion.
> Пример для двух зарегистрированных маршрутов Polza; другие провайдеры требуют
> собственного подтверждённого контракта и явного выбора владельца.

## Шаг 1: Установка

Установка использует сеть: агент выполняет её только после отдельного
разрешения владельца. Без разрешения отметь установку `NOT_RUN`, не запускай
`pip`/`uvx` для пробы.

```powershell
pip install "voiceover-pipeline[timing-whisper]"
```

## Шаг 2: Проверка окружения

`doctor` может прочитать приватный env-файл приложением ради проверки наличия
ключа и показать разрешённый путь, но не значение; агент файл не читает.
Запускай команду только в одобренном окружении. Без согласия проверь лишь
`voiceover help --json` и пометь key-preflight как `NOT_RUN`.

```powershell
voiceover doctor --provider <PROVIDER> --with-timings --json
```

Убедись что `workflow_ok: true`, а не только что exit code равен нулю.

## Шаг 3: Валидация сценария

```powershell
voiceover validate --script "script.md" --json
```

Если `valid: false` — покажи пользователю issues, не продолжай.

## Шаг 4: Генерация озвучки + таймингов

Обе команды ниже потенциально платные. До запуска владелец должен отдельно
разрешить именно выбранный provider/model и число POST с доказанной верхней
границей стоимости в согласованном бюджете; старые smoke-цены не годятся.
Без такого доказательства — `BLOCKED`, не пробный запрос. Локальная модель
Whisper для интегрированного `--with-timings` должна быть заранее установлена
и закеширована: preflight откажет до платного TTS, не скачивая веса неявно.
Отдельный standalone `timings` может скачать модель и требует разрешения на сеть.

Polza Chat Audio (chat-based, может добавить речь):

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
  --resume
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
  --resume
```

## Шаг 5: Чтение артефактов

```python
import json

# Только после успешного integrated generate --with-timings: без таймингов
# manifest не содержит timings_json/srt. Partial/failed результат не принимаем.
manifest = json.load(open("out/production/manifest.json"))
if not manifest.get("timings_json") or not manifest.get("srt"):
    raise RuntimeError("timings/srt absent: inspect JSON, do not guess paths")

# Точные тайминги в миллисекундах
timings = json.load(open(manifest["timings_json"]))

# Субтитры
srt_path = manifest["srt"]

# Чанки — пути/метаданные, не источник точной суммы; её смотри в history costs.
chunks = json.load(open(manifest["chunks_json"]))
```

## Шаг 6: Создание scene plan для Remotion

Whisper-сегменты мельче сцен: группируй сегменты по смысловым сценам.

```python
import json, re

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

manifest = json.load(open("out/production/manifest.json"))
if not manifest.get("timings_json"):
    raise RuntimeError("integrated timings missing; inspect CLI result")
timings = json.load(open(manifest["timings_json"]))

# Смысловые сцены из script.md (текст каждой сцены)
script_scenes = [
    {"title": "Вступление", "text": "Максимальное качество видео..."},
    {"title": "Бесплатные модели", "text": "Бесплатные модели..."},
    {"title": "Добавим провайдера", "text": "Добавим провайдера OpenAI..."},
]

scenes = []
seg_idx = 0
for scene in script_scenes:
    matched_segs = []
    scene_norm = normalize(scene["text"])
    # Собери сегменты, чей текст входит в сцену
    while seg_idx < len(timings["segments"]):
        seg = timings["segments"][seg_idx]
        if normalize(seg["text"]) in scene_norm:
            matched_segs.append(seg)
            seg_idx += 1
        else:
            break

    if matched_segs:
        scenes.append({
            "title": scene["title"],
            "start_ms": matched_segs[0]["start_ms"],
            "end_ms": matched_segs[-1]["end_ms"],
            "duration_ms": matched_segs[-1]["end_ms"] - matched_segs[0]["start_ms"],
            "narration": " ".join(s["text"] for s in matched_segs),
            "words": [w for s in matched_segs for w in s.get("words", [])]
        })
```

## Шаг 7: Использование в Remotion

- `scene["duration_ms"]` → `<Sequence durationInFrames={msToFrames(scene["duration_ms"])}>`
- `scene["words"]` → синхронизированная подсветка слов
- `srt_path` → `<Subtitles src={srt_path} />` или парсинг в кастомный компонент

## Важные правила

1. **НЕ оценивай длительность по словам.** Есть `.timings.json` → используй его.
2. **НЕ гадай имена файлов.** При успешных интегрированных таймингах проверь
   `manifest.json`; отдельный `voiceover timings --run-id <другой-id> --json`
   пишет в другой root, и его `files.timings_json`/`files.srt` берутся из
   ответа именно этой команды, а не из манифеста генерации.
3. **НЕ игнорируй exit codes.** Legacy timing может дать `40`, нативный partial
   результат — `50`; сперва проверь JSON/history и сохранённый MP3, не повторяй
   TTS и не скачивай модель без разрешения.
4. **НЕ перезаписывай платный output.** Используй безопасный `--resume` или
   `--skip-existing`; новая попытка требует отдельного решения владельца.
5. **Выбор провайдера** — читай `docs/05-providers-and-models.md`; актуальные
   зарегистрированные модели/голоса даёт `voiceover list ...`, но тариф и
   слышимое качество он не подтверждает.
6. **НЕ используй `chunks[].duration_ms` для длительности сцен.**
   Whisper-сегменты — источник истины. Границы чанков НЕ совпадают со смысловыми границами.
7. **Группируй Whisper-сегменты по смысловым сценам.**
   Одна сцена = несколько сегментов. `scene.durationInFrames` =
   интервал от первого до последнего сегмента сцены.
