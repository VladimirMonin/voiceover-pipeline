# Команды и флаги: полный CLI-справочник

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: все 8 команд, флаги, exit codes, JSON-контракт, правила безопасности.

## Команды

| Команда | Назначение | JSON |
|---|---|---|
| `doctor` | Проверить окружение (Python, FFmpeg, ключи, Whisper, CUDA) | Да |
| `validate --script` | Проверить Markdown-сценарий | Да |
| `list providers` | Показать доступных TTS-провайдеров | Да |
| `list voices --provider` | Показать голоса провайдера | Да |
| `list timing-models` | Показать Whisper-модели | Да |
| `split --script` | Разбить сценарий на чанки (без генерации) | Да |
| `generate` | Полная генерация: TTS + MP3 + опционально тайминги | Да |
| `timings --audio` | Извлечь Whisper-тайминги из готового MP3 | Да |

Все команды поддерживают `--json` для машинно-читаемого вывода.

## Exit codes

| Код | Значение | Когда |
|---:|---|---|
| 0 | success | Всё ок |
| 2 | invalid args | Файл не найден, неверный run-id, 0 чанков |
| 10 | missing dependency | faster-whisper не установлен |
| 11 | no ffmpeg/ffprobe | FFmpeg не найден в PATH |
| 20 | no key | Нет POLZA_API_KEY или OPENROUTER_API_KEY в .env |
| 30 | provider/run error | API error, папка существует без --overwrite |
| 40 | whisper error | Whisper timing не удался (но MP3 сохранён!) |
| 50 | output error | Ошибка записи/удаления файлов |

## Stdout/Stderr контракт

**С `--json`:**
- `stdout`: ровно один JSON object (success или error)
- `stderr`: progress-логи и предупреждения
- exit code: семантический

**Без `--json`:**
- `stdout`/`stderr`: человекочитаемый вывод
- `stderr`: ошибки и предупреждения

## JSON output contract

### Success

```json
{"status": "success", "..."}
```

### Error

```json
{"status": "error", "error": "описание", "code": 30}
```

## Команда `generate` — все флаги

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--provider` | choice | `polza-chat-audio` | `polza-chat-audio`, `polza-tts`, `openrouter-tts`, `qwen-local` |
| `--model` | str | `openai/gpt-audio-mini` | ID модели |
| `--script` | path | `in/script.md` | Путь к Markdown-сценарию |
| `--delimiter` | str | `******` | Разделитель чанков |
| `--output-dir` | path | `out` | Корень выходной директории |
| `--run-id` | str | авто | Имя прогона (только `[a-zA-Z0-9._-]`) |
| `--voice` | str | зависит от провайдера | Голос |
| `--fallback-voice` | str | `onyx` | Запасной голос для Polza Chat Audio |
| `--style-prompt` | str | дефолтный | Стиль подачи для TTS (OpenRouter Gemini) |
| `--style-prompt-file` | path | — | Читать prompt из файла |
| `--no-style-prompt` | flag | false | Отключить prompt полностью |
| `--no-trim` | flag | false | Не обрезать финальную тишину |
| `--json` | flag | false | JSON-вывод в stdout |
| `--overwrite` | flag | false | Удалить существующую папку прогона |
| `--skip-existing` | flag | false | Пропустить если прогон уже есть |

### Qwen-local опции

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--mode` | choice | `preset` | `preset` (готовый голос) или `clone` (клонирование) |
| `--sample` | str | — | Путь к референс-аудио для clone |
| `--sample-text` | str | `""` | Текст референса для clone (точнее) |

### Whisper timing опции (generate + timings)

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--with-timings` | flag | false | Запустить Whisper после TTS |
| `--timing-model` | choice | `small` | `base`, `small`, `medium`, `large-v3-turbo`, `large-v3` |
| `--timing-device` | choice | `cpu` | `auto`, `cpu`, `cuda` |
| `--timing-compute` | choice | `int8` | `auto`, `int8`, `int8_float16`, `float16`, `float32` |
| `--timing-language` | str | `ru` | Код языка для Whisper |
| `--word-timestamps` | flag | false | Добавить word-level тайминги |

## Команда `timings` — флаги

| Флаг | Тип | Default | Назначение |
|---|---|---|---|
| `--audio` | str | **обязательный** | Путь к MP3-файлу |
| `--output-dir` | path | `out` | Корень выходной директории |
| `--run-id` | str | stem аудиофайла | Имя прогона |
| `--model` | choice | `small` | Whisper-модель |
| `--device` | choice | `cpu` | `auto`, `cpu`, `cuda` |
| `--compute` | choice | `int8` | Тип вычислений |
| `--language` | str | `ru` | Код языка |
| `--json` | flag | false | JSON-вывод |
| `--word-timestamps` | flag | false | Word-level тайминги |
| `--overwrite` | flag | false | Перезаписать |
| `--skip-existing` | flag | false | Пропустить |

## `list voices` — JSON контракт

```powershell
voiceover list voices --provider polza-tts --json
```

Ответ:

```json
{
  "status": "success",
  "provider": "polza-tts",
  "voices": ["alloy", "ash", "ballad", "coral", ...],
  "voice_categories": {
    "openai": ["alloy", "ash", "ballad", ...],
    "elevenlabs": ["Rachel", "Aria", "Roger", ...]
  }
}
```

- `voices` — **всегда** плоский массив (backward-compatible)
- `voice_categories` — объект с разбивкой по семействам (опционально, есть у `polza-tts` и `openrouter-tts`)

Для `openrouter-tts` категории: `"gemini"` и `"openai"`.

## `--run-id` правила

Разрешено: `[a-zA-Z0-9._-]`, например `prod`, `prod-01`, `prod_01`, `prod.v1`.

Запрещено:
- `.`, `..`, путь с `/` или `\`
- Leading/trailing whitespace
- Trailing dot или space
- Абсолютные пути
- Windows reserved names: `CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`
- Illegal chars: `<>:"|?*` и control chars

## `--output-dir` правила

Запрещено:
- Drive root (`C:\`)
- Home directory
- CWD (current working directory)

Разрешено: относительные (`out`, `out/project`) и абсолютные пути вне CWD/home/root.

## Existing output policy

| Ситуация | Поведение |
|---|---|
| Папка не существует | Создать |
| Папка существует + `--overwrite` | Удалить папку полностью, создать заново |
| Папка существует + `--skip-existing` | Вернуть `status: skipped`, не менять файлы |
| Папка существует без флагов | Ошибка exit code 30 |

## Safe defaults

| Параметр | Default | Почему |
|---|---|---|
| `--timing-device cpu` | CPU | Всегда работает |
| `--timing-compute int8` | INT8 | Минимальный RAM |
| `--timing-model small` | 486 MB | Минимальный для русского |
| Default: no overwrite | Ошибка | Защита от случайной перезаписи |
