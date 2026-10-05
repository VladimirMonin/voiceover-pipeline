# Рабочие сценарии (workflows)

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ.
> Здесь: готовые end-to-end цепочки для типовых задач.
> Канонические команды и офлайн-границы — `voiceover help start.quick` / `runs.resume`;
> текущие модели и голоса — `voiceover list ...`, курируемый обзор — `docs/05-providers-and-models.md`.
> Все команды `doctor` могут проверить ключ через env-файл приложения: запускайте
> их только в одобренном окружении; сам агент файл не читает. Каждый live/provider
> GET/POST, сетевая установка и загрузка модели требуют отдельного разрешения
> владельца и применимого доказанного потолка для платного вызова. Примеры ниже
> не являются таким разрешением.

## Fresh Project Bootstrap (нет ничего)

Когда проект пустой — ни `.env`, ни `script.md`, ни `out/`:

1. **Установи пререквизиты.** Проверь Python/UV/FFmpeg, если нет — поставь сам.
   Если среда не позволяет — попроси пользователя (см. `docs/02-install.md`).
   Сетевые установки и загрузку моделей выполняй только с явного разрешения владельца.
   Если на любом шаге команда не найдена — `docs/09-troubleshooting.md`.
2. **Выбери сборку.** Production → `voiceover-pipeline[timing-whisper]`.
   Запусти без установки: `uvx voiceover-pipeline doctor` или установи постоянно: `pipx install "voiceover-pipeline[timing-whisper]"`.
3. **Создай `.env.example`.** Из шаблона `examples/env-example.md` (только placeholder-ы).
4. **Создай `.gitignore`.** Если файла нет — создай. Добавь строку `.env`.
5. **Создай `script.md`.** Если сценария нет — создай из шаблона `examples/minimal-script.md`.
6. **Создай `out/`.** Папка для артефактов, будет использована как `--output-dir`.
7. **Ключи — только владелец.** Агент НЕ создаёт, не копирует и не читает реальный `.env`.
   Попроси владельца ОДИН раз поместить ключи в приватный env-файл вне инструментов агента
   или выставить их в окружении процесса (см. `docs/03-security-and-secrets.md`).
   В одобренном окружении проверь наличие ключа через
   `voiceover doctor --provider <X> --json` (путь, не значение). Больше не спрашивать.
8. **Дальше — выбери workflow ниже.**

## Golden Cloud Workflow — Polza Chat Audio (рубли, chat-based)

```powershell
voiceover doctor --provider polza-chat-audio --with-timings --json
voiceover validate --script "script.md" --json
voiceover generate `
  --provider polza-chat-audio `
  --model "openai/gpt-audio-mini" `
  --script "script.md" `
  --run-id "prod" `
  --json `
  --resume
voiceover timings --audio "out/prod/prod-voiceover-openai-gpt-audio-mini.mp3" --run-id "prod-timings" --json
```

`prod-timings` — отдельный root: возьми `files.timings_json` и `files.srt`
из JSON-ответа именно `timings`, а не из `out/prod/manifest.json`. Если нужны
пути к таймингам в манифесте генерации, выбирай интегрированный
`generate --with-timings` заранее: локальная модель должна быть в кеше до
платного TTS; отдельная команда `timings` может скачать веса и требует
разрешения на сеть.

## Golden Cloud Workflow — Polza TTS (рубли, классический TTS)

OpenAI TTS через Polza — JSON base64, быстро:

```powershell
voiceover doctor --provider polza-tts --with-timings --json
voiceover validate --script "script.md" --json
voiceover generate `
  --provider polza-tts `
  --model "openai/gpt-4o-mini-tts" `
  --voice "ash" `
  --script "script.md" `
  --run-id "prod" `
  --json `
  --resume
```

ElevenLabs через Polza — async `/media`, чистое качество:

```powershell
voiceover generate `
  --provider polza-tts `
  --model "elevenlabs/text-to-speech-turbo-2-5" `
  --voice "Rachel" `
  --script "script.md" `
  --run-id "elevenlabs" `
  --json `
  --resume
```

## Golden Cloud Workflow — OpenRouter (доллары)

Gemini TTS — западные голоса через `/audio/speech`:

```powershell
voiceover generate `
  --provider openrouter-tts `
  --model "google/gemini-3.1-flash-tts-preview" `
  --voice "Puck" `
  --script "script.md" `
  --run-id "gemini-prod" `
  --json `
  --resume
```

- OpenRouter принимает только top-level `voice`; `input` byte-equals произносимому
  тексту реплики. Явные `--style-prompt`/`--style-prompt-file` отклоняются до
  платного запроса, а `--no-style-prompt` — совместимый no-op.
- `google/gemini-3.8-flash-tts`/`google/gemini-3.8-flash-lite-tts` — обычные
  speech-модели этого маршрута (и `polza-tts`) без opt-in: один scalar `voice`
  на запрос, направление части — отдельным `instructions`.
- Для этого OpenRouter-маршрута только произносимый turn text (включая допустимые
  inline audio tags) попадает в `input`; метаданные `vibe`/profile не становятся
  отдельной инструкцией провайдеру. Не обещайте эффект режиссуры без live/listening
  приёмки (см. `docs/11-gemini-prompting.md`).

## Qwen Local Workflow (без тарифа API, GPU)

Требуется NVIDIA GPU + CUDA + extras `voiceover-qwen`:

```powershell
voiceover doctor --provider qwen-local --json
voiceover generate `
  --provider qwen-local `
  --mode preset `
  --voice "Aiden" `
  --script "script.md" `
  --run-id "qwen-prod" `
  --json `
  --resume
```

Clone-голос (нужен референс-аудиофайл):

```powershell
voiceover generate `
  --provider qwen-local `
  --mode clone `
  --sample "my_voice_sample.mp3" `
  --sample-text "Текст референса." `
  --script "script.md" `
  --run-id "my-voice" `
  --json `
  --resume
```

## Agent Podcast Workflow (dialogue)

Канонический путь для «сделай подкаст с двумя ведущими».

> OpenRouter использует один документированный top-level `voice` на turn;
> `multi_speaker_voice_config` не отправляется. Offline contract реализует
> turn-by-turn concat с детерминированными паузами. Не заявляй audible PASS
> без отдельного human listening acceptance для provider.

1. **Автор сценарий.** Из темы и состава ведущих пользователя напиши
   `podcast.md` в формате `dialogue` (см. `docs/04-input-format.md`,
   пример `examples/gemini-dialogue-podcast.md`). Ровно два спикера, два
   различных голоса.
2. **Валидация без платного запроса:**
   ```powershell
   voiceover validate --script "podcast.md" --format dialogue --agent --json
   ```
3. **Проверка окружения** (приложение может проверить ключ/env-файл; агент его не читает; только в одобренном окружении):
   ```powershell
   voiceover doctor --provider openrouter-tts --json
   ```
4. **Спроси разрешение** на платную генерацию, если оно ещё не дано.
5. **Генерация:**
   ```powershell
   voiceover generate --script "podcast.md" --run-id "podcast-prod" --json --resume
   ```
   Provider/model/каст/направление берутся из frontmatter; явные флаги
   допустимы, но самодостаточный скрипт предпочтителен.
6. **Прочитай артефакты:** `manifest.json` (entry-point), `run_state.json`,
   `generation.log`.
7. **Тайминги:** если нужны — предпочитай `generate --with-timings` в том же
   безопасном прогоне. Отдельный `voiceover timings` — только с ДРУГИМ
   `--output-dir`/`--run-id`, никогда с `--overwrite` по папке платного прогона.

## OmniVoice Local Workflows (без тарифа API, GPU)

Требуется NVIDIA GPU + CUDA + native audio.cpp package
(см. `docs/14-local-audio-cpp-models.md`). Обычная озвучка использует один
голос и один native session; `dialogue` маршрутизирует отдельный bank profile на turn.

Дефолтный preset из voice bank (без `--voice` берётся `default_voice` каталога):

```powershell
voiceover doctor --provider omnivoice-local --json
voiceover generate `
  --provider omnivoice-local `
  --mode preset `
  --voice-bank "C:\audio-cpp-work\voice-bank\approved\catalog.json" `
  --script "script.md" `
  --run-id "omni-bank" `
  --json `
  --resume
```

Именованный профиль банка:

```powershell
voiceover generate `
  --provider omnivoice-local `
  --mode preset `
  --voice-bank "C:\audio-cpp-work\voice-bank\approved\catalog.json" `
  --voice "omni-male-neutral-01" `
  --script "script.md" `
  --run-id "omni-male" `
  --json `
  --resume
```

Ad-hoc clone:

```powershell
voiceover generate `
  --provider omnivoice-local `
  --mode clone `
  --reference-audio "my_voice.wav" `
  --reference-text "Текст референса." `
  --script "script.md" `
  --run-id "omni-clone" `
  --json `
  --resume
```

Design-инструкция:

```powershell
voiceover generate `
  --provider omnivoice-local `
  --mode design `
  --design-instruction "female, young adult, moderate pitch" `
  --script "script.md" `
  --run-id "omni-design" `
  --json `
  --resume
```

Auto-голос (без voice guidance):

```powershell
voiceover generate `
  --provider omnivoice-local `
  --mode auto `
  --script "script.md" `
  --run-id "omni-auto" `
  --json `
  --resume
```

## Timings из готового MP3

Когда MP3 уже есть, а нужны только тайминги:

```powershell
voiceover timings `
  --audio "path/to/audio.mp3" `
  --output-dir "out" `
  --run-id "timed-audio" `
  --model small `
  --device cpu `
  --compute int8 `
  --language ru `
  --word-timestamps `
  --json
```

Если output уже существует, выберите новый `--run-id`; не удаляйте чужие
артефакты по умолчанию.

## Безопасный повторный запуск

```powershell
voiceover generate ... --resume            # продолжить безопасно
voiceover generate ... --skip-existing     # пропустить если есть
voiceover generate ... --run-id "prod-02"  # новый run-id
voiceover generate ... --overwrite --confirm-delete-paid-audio  # только по явному решению владельца о потере и цене
```

## Интеграция с Remotion

Полный поток «сценарий → сцены» (код `scene plan`, чтение `manifest.json`/`.timings.json`,
группировка Whisper-сегментов по смысловым сценам) — в примере
[examples/remotion-agent-flow.md](../examples/remotion-agent-flow.md).

- `manifest.json` — entry-point генерации; `timings_json`/`srt` в нём есть
  только при успешных интегрированных таймингах в том же root. Для standalone
  `timings` читай `files.timings_json`/`files.srt` из его JSON-ответа.
- Если `.timings.json` действительно получен, используй его durations, а не
  words-per-second и не `chunks[].duration_ms`.
- Платный прогон не перезаписывай `--overwrite`; оборванный прогон продолжай через `--resume`.

## Выбор провайдера

Канонично — `voiceover help providers.polza`, `voiceover list providers --json` и
`voiceover list voices --provider <X> --json`; тариф эти команды не подтверждают.
Краткий обзор классов маршрутов и их ограничений —
[docs/05-providers-and-models.md](05-providers-and-models.md); старые цены не
используйте для выбора или бюджета.
