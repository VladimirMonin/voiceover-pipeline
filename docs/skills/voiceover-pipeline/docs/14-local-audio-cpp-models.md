# Локальные модели и hybrid `audio.cpp` runtime

> АГЕНТ: ЧИТАЙ ЭТОТ ФАЙЛ ЦЕЛИКОМ перед выбором Qwen3-ASR, Nemotron,
> Qwen3-TTS или OmniVoice, сравнением скорости либо заявлением о Windows.

## Что входит в локальный контур

| Семейство | Задача | Provider ID | Маршруты |
|---|---|---|---|
| Qwen3-ASR | распознавание + contextual text | `qwen-local` | Python rollback; explicit `audio.cpp` |
| Qwen Forced Aligner | уточнение границ | внутренний Qwen route | отдельная модель |
| Nemotron 3.5 ASR | быстрое локальное распознавание | `nemotron-local` | Python rollback; explicit `audio.cpp` |
| Qwen3-TTS | preset, cloning, voice design | `qwen-local` | Python default; explicit `audio.cpp` |
| OmniVoice | offline built-in female style condition | `omnivoice-local` | explicit `audio.cpp` |

Faster-Whisper остаётся отдельным runtime. Облачные провайдеры не подменяются
локальными. Выбор `audio.cpp` должен быть явным и fail-closed.

## Как читать скорость

RTF — отношение времени вычисления к длительности аудио:

- RTF меньше `1` означает быстрее реального времени;
- `RTF × 60` даёт секунды вычисления на минуту аудио;
- `1 / RTF` даёт кратность быстрее реального времени.

Не сравнивать RTF разных машин, корпусов, warm/cold режимов или runtimes как
универсальный рейтинг модели.

## Принятый ASR benchmark

Источник: WVM corpus `wvm-slice5-local-reference-v1`, 50 запусковых случаев,
примерно 505,73 секунды аудио в каждом режиме. Качество обычной речи считается
по 18 случаям со state `non_empty`. Полные 50 включают no-speech, diagnostic и
filtered/rejected cases; их общий WER/CER — execution total, а не чистая оценка
качества речи.

### Qwen3-ASR без contextual text

- WER по 18 речевым случаям: `33,10%`;
- CER: `16,18%`;
- RTF: `0,13933`;
- около `8,36` секунды вычисления на минуту аудио;
- около `7,18×` быстрее реального времени;
- wall time всего корпуса: `70,47` секунды;
- peak VRAM: `2,462 GiB`;
- peak RAM: `2,78 GiB`.

### Qwen3-ASR с contextual text

- WER по 18 речевым случаям: `30,28%`;
- CER: `15,52%`;
- RTF: `0,11078`;
- около `6,65` секунды вычисления на минуту аудио;
- около `9,03×` быстрее реального времени;
- wall time всего корпуса: `56,03` секунды;
- peak VRAM: `2,685 GiB`;
- peak RAM: `2,78 GiB`.

Prompt-on выполнялся вторым в одном прогретом процессе. Снижение WER/CER —
наблюдаемый результат корпуса, но ускорение нельзя приписывать contextual text:
оно смешано с warm-run эффектом.

### Nemotron 3.5 ASR

- WER по 18 речевым случаям: `32,39%`;
- CER: `14,72%`;
- RTF: `0,06230`;
- около `3,74` секунды вычисления на минуту аудио;
- около `16,05×` быстрее реального времени;
- wall time всего корпуса: `31,51` секунды;
- peak VRAM: `4,556 GiB`;
- peak RAM: `3,268 GiB`.

На этом benchmark Nemotron быстрее. Qwen экономнее по VRAM и единственный из
двух принятых runs с contextual text matrix. По качеству нет абсолютного
победителя: Qwen с контекстом лучше по WER, Nemotron лучше по CER; речевой срез
содержит только 18 случаев.

## Таймкоды и граница доказательств

Практический ASR contract проекта: распознанный текст плюс phrase/segment
 timings; word timings дополнительны. Однако приведённый Python benchmark не
выдал timestamp boundaries. Он доказывает скорость и текстовые ошибки, но не
качество alignment.

Qwen `audio.cpp` route использует отдельный Forced Aligner. Nemotron route
сохраняет model-native timing data и нормализует tokens-to-words. Заявлять
таймкоды принятыми можно только после live evidence конкретного route, а не по
наличию кода или static tests.

## Скорость синтеза речи

### OmniVoice accepted live run

Русский WAV:

- длительность речи: `83,38` секунды;
- generation wall time: `68,82` секунды;
- RTF: `0,825`;
- `49,52` секунды вычисления на минуту готовой речи;
- около `1,21×` быстрее реального времени;
- PCM signed 16-bit, mono, `24 kHz`.

Это локальный noncommercial run весов CC-BY-NC. Он не разрешает распространять
веса или обещать commercial availability.

### Qwen3-TTS Python baseline

Отдельный реальный baseline этой машины, Qwen3-TTS 1.7B CustomVoice, Sohee:

- длительность речи: `51,408` секунды;
- полный wall time с загрузкой модели: `171,71` секунды;
- RTF: около `3,34`;
- около `200,41` секунды вычисления на минуту готовой речи;
- примерно `0,30×` real-time, то есть около трёх минут двадцати секунд
  вычислений на минуту речи.

Этот baseline относится к Python route и включает cold model load. Не выдавать
его за скорость нового `audio.cpp` Qwen3-TTS route: для него нужен отдельный
live measurement.

### Qwen3-TTS Linux container config

`audio.cpp` выбирается только явным `VOICEOVER_QWEN_TTS_RUNTIME=audio-cpp`.
Указать `VOICEOVER_AUDIO_CPP_QWEN_TTS_MODEL` нужно на локальную директорию
Safetensors-пакета с `model.safetensors`, `config.json`, `tokenizer_config.json`
и поддиректорией `speech_tokenizer`; VOP ничего не скачивает и не требует
искусственный GGUF-файл. Опциональный
`VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON` — JSON argv для локального Docker
command, default `["docker"]`; shell-like строка недопустима.

Pinned Linux container запускается без сети, с read-only root, ограниченным
`/tmp`, GPU и отдельной private output directory. CustomVoice передаёт
`speaker` и `instruct` в task `tts`; Base clone — readonly WAV и reference text
в `clon`; VoiceDesign — `instruct` в `vdes`. Это contract/offline evidence, не
live claim: для каждого варианта нужен отдельно установленный package и
разрешённый реальный GPU run.

## Числительные для локального TTS

До любого локального production-run произносимый текст должен быть отдельной
human-readable версией:

- цифры, проценты, версии и дроби записаны словами в нужном падеже;
- аббревиатуры раскрыты или записаны так, чтобы голос произнёс их ожидаемо;
- ID, SHA, пути и JSON не читаются вслух;
- validator warning `contains_digits` устранён, а не проигнорирован.

Machine-readable receipt сохраняет точные числа отдельно. Не заменять им
произносимый сценарий.

## Linux и Windows

Linux production route использует pinned CUDA container и exact source/image
revision. Native Windows использует `audiocpp_cli.exe`, colocated DLL/package,
Windows process-group cancellation и SHA checks. Docker, WSL и Wine не являются
native Windows evidence.

Static Windows tests подтверждают код, но не реальный Windows 10/11 x64 run.
Для live acceptance нужен настоящий Windows host с NVIDIA GPU и native package.

## Перед заявлением «готово»

1. Указать конкретный runtime: Python, Linux container или native Windows.
2. Разделить static/offline tests и live inference evidence.
3. Для benchmark назвать corpus, population и warm/cold boundary.
4. Для TTS сохранить WAV, receipt, elapsed, duration и RTF.
5. Проверить cleanup GPU/process/temp files.
6. Не повторять model run только ради документационной правки.
