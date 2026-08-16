# Оценка применимости audio.cpp в Voiceover Pipeline

> **Уточнение владельца от 2026-08-16:** отказ от полной миграции остаётся в силе, но целевая гибридная схема теперь рассматривает перенос всех локальных non-Whisper ASR/TTS families на общий runtime. См. [датированное addendum](2026-08-16-audio-cpp-hybrid-consolidation.md).

**Дата среза:** 2026-08-16
**Voiceover Pipeline:** commit `f3cb5db77628221ef7750cd985b42158c574e1b9`
**audio.cpp:** `main` commit `502b5b74bd26e9b4aed267d1776ecf131cae7215`, время commit `2026-08-15T19:10:16Z`.[1]

## 1. Краткий вердикт

| Вариант | Решение |
|---|---|
| Полная миграция VOP на audio.cpp | **REJECT** |
| audio.cpp как optional backend | **MODIFY / кандидат после spike** |
| Изолированный standalone spike | **ACCEPT** |
| Оставить production без изменений | **ACCEPT сейчас** |

**Рекомендация:** не заменять Faster-Whisper, текущие Qwen/Nemotron Python adapters или облачные providers. Сначала выполнить отдельный, не входящий в production-код spike на тех же 50 WVM Slice 5 assets. Если relative quality, resource, reproducibility и machine-contract gates пройдены, добавить `audio.cpp` только как тонкий optional ASR provider через существующий `ASRProviderRegistry`.

Главный блокер полной миграции: в audio.cpp есть core-реализации Qwen3-ASR, Nemotron 3.5 ASR и Qwen3-TTS, но нет самостоятельной Whisper/Faster-Whisper model family. Вхождения Whisper frontend внутри других моделей не являются заменой Faster-Whisper. Кроме того, интерфейс `audiocpp_cli` не совпадает с JSON/exit-code/receipt/artifact контрактами VOP.

## 2. Текущее состояние Voiceover Pipeline

VOP уже имеет правильный integration seam:

- `ASRProvider` в `src/voiceover_pipeline/providers/base.py`;
- typed `ASRRequest`, `ASRResult`, `ASRCapabilities`, `ASRExecutionReceipt`, segments/words/hints в `models.py`;
- `ASRProviderSpec` и `ASRProviderRegistry` в `providers/asr_registry.py`;
- generic `transcribe_cmd` в `cli.py`, который разрешает provider через registry, проверяет dependency/capabilities и контролирует returned provider ID, timestamps и alignment origin;
- Python adapters `qwen-local` и `nemotron-local`;
- отдельный incumbent `FasterWhisperProvider`;
- облачные TTS и transcription providers;
- стабильный machine contract: один JSON object в stdout, diagnostics в stderr, semantic exit codes `0/2/10/11/20/30/40/50`;
- стандартная карта generation/timing artifacts и fail-closed behavior.

Qwen3-ASR Python backend реально проверен на 50 WVM Slice 5 cases с prompt off/on. Принятый evidence directory:

`out/asr-benchmarks/l1-qwen3-asr-0.6b-t_73a7dd71-rerun1/`

Он фиксирует модель `Qwen/Qwen3-ASR-0.6B`, cache revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, CUDA/bfloat16 и 50 cases на каждый prompt mode. Full offline baseline после Qwen: **172 passed, 1 skipped**. На момент запуска аудита Nemotron benchmark ещё выполнялся. Позже он завершился и вошёл в основной ASR acceptance, но его результат не использовался как runtime-доказательство пригодности audio.cpp: для этого по-прежнему нужен отдельный spike.

### 2.1. Доказательство, что VOP сейчас не использует audio.cpp

Проверка выполнена обязательным маршрутом Codebase → Serena → ast-grep.

**Codebase project:** `home-v-code-voiceover-pipeline`.

Точные запросы:

1. `audio.cpp|audiocpp|audio_cpp`, scope
   `^(src/voiceover_pipeline|pyproject\.toml|uv\.lock)`
   → `total_grep_matches=0`, `total_results=0`.
2. `ASRProviderRegistry|ASRProviderSpec|ASRRequest|ASRExecutionReceipt`, scope `^src/voiceover_pipeline`
   → owners: `providers/asr_registry.py`, `models.py`, `cli.py`, Qwen/Nemotron adapters.
3. `trace_path(transcribe_cmd, outbound, depth=3)`
   → CLI validation/result helpers и provider transcription path; audio.cpp edge отсутствует.
4. `trace_path(build_provider, outbound, depth=3)`
   → Polza, OpenRouter и `QwenLocalTTSProvider`; audio.cpp edge отсутствует. Этот polymorphic graph сам по себе не доказывает runtime target, поэтому он был перепроверен символами и structural matches.

**Serena symbols:**

- `ASRProviderSpec`;
- `transcribe_cmd`;
- `QwenLocalASRProvider/transcribe`;
- `NemotronLocalASRProvider/transcribe`;
- `FasterWhisperProvider/transcribe`;
- `QwenLocalTTSProvider/_load_model`;
- references к `ASRProviderSpec`.

Результат: Qwen ASR вызывает `qwen_asr.Qwen3ASRModel`; Nemotron — `transformers.AutoModelForRNNT`; Qwen TTS — `qwen_tts.Qwen3TTSModel`; timings — `faster_whisper.WhisperModel`. Ни один owner не импортирует и не запускает audio.cpp.

**ast-grep patterns:**

- `ASRProviderSpec($$$ARGS)` → ровно 2 production matches: Qwen и Nemotron;
- `class $C(ASRProvider): $$$BODY` → ровно 2 production implementations: Qwen и Nemotron;
- `from qwen_tts import Qwen3TTSModel` → 1 match;
- `from faster_whisper import WhisperModel` → 1 match;
- `subprocess.run($$$ARGS)` → 9 production matches, относящиеся к FFmpeg/FFprobe и resource sampling через `nvidia-smi`; вызова `audiocpp_cli` нет.

Дополнительно `pyproject.toml` содержит optional dependencies для Faster-Whisper, Qwen ASR и Nemotron/Transformers, но не audio.cpp. Следовательно, утверждение **«VOP сейчас не использует audio.cpp» подтверждено статически для commit `f3cb5db...`**.

## 3. Фактическая поддержка audio.cpp

Кандидаты находятся в `src/models`, а не в `community_models`, и добавлены в default build registry через `CMakeLists.txt`; поэтому это **core**, а не community support.[3]

| Семейство/вариант | Статус | Точные anchors | Итог |
|---|---|---|---|
| Qwen3-ASR-0.6B | **core** | `model_specs_v1/qwen3_asr.json`: GGUF Q8_0/F16 и HF Safetensors package; `src/models/qwen3_asr/loader.cpp`; `session.cpp`.[4][5][6] | Поддерживается |
| Qwen3-ASR-1.7B | **core** | тот же spec: GGUF Q8_0/F16 и HF Safetensors; тот же loader/session.[4][5][6] | Поддерживается |
| Nemotron 3.5 ASR Streaming 0.6B | **core** | `model_specs_v1/nemotron_asr.json`: Q8_0/F16/Safetensors; `loader.cpp`; offline и streaming sessions.[7][8][9] | Поддерживается |
| Qwen3-TTS Base | **core** | `qwen3_tts.json`; variant-specific loader/session; 0.6B Base Safetensors и 1.7B packages.[10][11][12] | Поддерживается |
| Qwen3-TTS CustomVoice | **core** | 1.7B Q8_0/BF16 packages; `speaker` и `instruction`; offline session.[10][11][12] | Поддерживается |
| Qwen3-TTS VoiceDesign | **core** | 1.7B Q8_0/BF16 packages; VoiceDesign task и `instruction`; offline session.[10][11][12] | Поддерживается |
| Whisper ASR | **absent** | нет Whisper loader, package spec или session; найденные Whisper frontends принадлежат Qwen/Higgs/SeedVC internals | Не заменяет Whisper |
| Faster-Whisper | **absent** | Python/CTranslate2 runtime не является model family audio.cpp | Не поддерживается |
| Указанные семейства как community | **absent** | они уже находятся в core tree | Не применимо |

## 4. Family-specific capabilities

### 4.1. Qwen3-ASR

- **Batch:** да.
- **Streaming:** loader и session реализуют offline и streaming, включая audio chunks, partial text и final result.[5][6]
- **Русский:** `ru` есть в model spec.[4]
- **Context:** `TaskRequest.text_input.text` передаётся в `Qwen3ASRRequest.context`. Это contextual prompt, но не доказательство glossary или weighted phrase boosting.[6]
- **Language:** current `main` дополнительно исправляет propagation `--language` для audio-only request.[1]
- **Phrase boosting/glossary:** отдельного подтверждённого API нет.
- **Word timestamps:** не нативный универсальный ASR output. `return_timestamps=true` требует отдельный `forced_aligner_path` и Qwen3 Forced Aligner; spec прямо описывает эту зависимость.[4]
- **Segments:** model session формирует transcript и optional words; ASR speech segments не подтверждены.
- **Confidence:** полезная model-specific confidence не подтверждена.
- **Long audio/VAD:** есть fixed/VAD/none/auto chunk modes; при timestamp mode используется VAD и отдельный Silero VAD path.[4][6]
- **Несогласованность upstream:** spec перечисляет только `offline`, тогда как loader/session и README заявляют streaming. Для production capability нужно доверять executable session и отдельно включить эту inconsistency в spike.

### 4.2. Nemotron 3.5 ASR

- **Batch:** да.
- **Streaming:** да; отдельная streaming session принимает chunks, публикует text deltas callback и возвращает final transcript.[8][9]
- **Русский:** `ru-RU` есть в spec.[7]
- **Prompt/context:** `prompt_dictionary` выбирает language/locale prompt ID. Это не arbitrary context, glossary или phrase boosting.[7][9]
- **Word timestamps:** session переносит `decoded.token_timestamps` в generic `word_timestamps`.[9]
- **Segments:** не подтверждены.
- **Confidence:** generic `WordTimestamp.confidence` существует, но model-specific meaningful confidence assignment не найден; считать поддержку confidence нельзя.
- **Long audio/VAD:** streaming даёт bounded chunk execution; встроенный VAD для этой family не подтверждён.
- **Partial result:** есть text delta callbacks, но streaming policy указывает final result; точную семантику revisions/order требуется проверить runtime.

### 4.3. Qwen3-TTS

- **Base:** offline TTS с reference audio и optional `reference_text`.
- **CustomVoice:** offline TTS, named `speaker`, optional natural-language `instruction`.
- **VoiceDesign:** отдельный offline VoiceDesign task, управляемый `instruction`.
- **Русский:** поддерживается.
- **Batch:** общий CLI имеет request sequence/batch inputs; внутри family есть long-text chunking.
- **Streaming:** **нет в audio.cpp Qwen3-TTS session** — loader допускает только offline, хотя official Qwen weights описывают streaming capability модели.[11][12][22] Нельзя переносить capability weights на конкретный runtime.
- **Long text:** `text_chunk_size`, default 8192 codepoints.[10][12]

### 4.4. Почему `--words-out` ничего не гарантирует

`audiocpp_cli` показывает `--segments-out`, `--turns-out` и `--words-out` как **common output options** для всех выбранных families.[14] Generic `TaskResult` действительно имеет `speech_segments`, `speaker_turns` и `word_timestamps`. Но `emit_task_result` пишет words только если `result.word_timestamps` не пуст.[15]

Следовательно:

- наличие `--words-out` доказывает только способность framework сериализовать words;
- generic confidence field доказывает только наличие поля;
- поддержка timestamps/confidence должна подтверждаться конкретными loader/session и runtime output каждой family;
- для Qwen timestamps зависят от forced aligner;
- для Nemotron words формируются decoder path;
- для Whisper ничего не формируется, потому что Whisper family отсутствует.

## 5. Совместимость с контрактами VOP

| Контракт VOP | audio.cpp напрямую | Требуемая адаптация |
|---|---|---|
| Один JSON object в stdout | нет; документированный CLI пишет `key=value`, transcript и metrics.[13][14] | provider читает files, возвращает VOP JSON через `transcribe_cmd` |
| Semantic exit codes | только `0/1` в `audiocpp_cli`.[14] | map missing binary/model → `10`, bad request → `2`, runtime → `30` |
| Typed request/result | собственный C++ `TaskRequest/TaskResult` | explicit mapping, без global capability assumptions |
| Execution receipt | нет VOP-compatible receipt | заполнить runtime SHA/version, model/package digest, device, compute, measurements |
| Registry/listing | собственный model registry | зарегистрировать один `ASRProviderSpec` |
| Doctor | нет VOP workflow health | dependency probe без inference |
| Artifacts | WAV/text/words files | VOP остаётся owner итоговых JSON/SRT/manifest artifacts |
| Prompt semantics | family-specific `--text`/language/instruction | Qwen context и Nemotron locale не смешивать |
| Cloud providers | отсутствуют | не трогать Polza/OpenRouter/Groq/xAI routes |
| Faster-Whisper timings | отсутствуют | сохранить incumbent без изменений |
| Qwen TTS MP3/chunk workflow | неэквивалентный offline WAV contract | не включать в первый provider |

Первый integration slice должен быть **ASR-only**. TTS через audio.cpp — отдельное последующее решение: оно затрагивает chunk synthesis, WAV→MP3 conversion, voice/reference semantics, resumability и generation artifacts и не должно скрываться внутри ASR spike.

## 6. CUDA, сборка и доставка моделей

Upstream Linux build требует GCC 13+, CMake и CUDA Toolkit 12+; для RTX 3000 документирован `CMAKE_CUDA_ARCHITECTURES=86`.[16]

```bash
cmake -S /path/to/audio.cpp -B /path/to/build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENGINE_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS="qwen3_asr;nemotron_asr"

cmake --build /path/to/build --parallel 4 --target audiocpp_cli
```

Это подтверждает **toolchain feasibility** для RTX 3060, но не доказывает runtime fit в 12 GB. Qwen Python benchmark на этой машине потреблял примерно 2.6–2.9 GB peak VRAM, однако это другой runtime и формат весов. Для audio.cpp Qwen/Nemotron/TTS peak VRAM, OOM headroom и multi-model coexistence — **не подтверждено** до spike.

Поддерживаются family-specific GGUF и Safetensors packages.[4][7][10] GGUF не является универсальным: arbitrary llama.cpp/whisper.cpp GGUF нельзя считать совместимым без нужных tensor names, metadata и package spec. Default package download revisions указывают на mutable `main`; для воспроизводимости spike обязан фиксировать:

- audio.cpp commit;
- compiler/CMake/CUDA versions и build flags;
- hash бинарника;
- package ID, source repo/revision;
- hash каждого weight/config/tokenizer файла;
- backend/device/options.

Model management пока не полностью native: открытый issue предлагает перенести управление моделями из Python в C++/`hf` CLI.[25] Поэтому production provider не должен автоматически скачивать или конвертировать веса.

## 7. Лицензии

Лицензии framework и model weights должны учитываться отдельно:

| Объект | Зафиксированный metadata |
|---|---|
| audio.cpp source | repository `LICENSE`: Apache License 2.0.[20] |
| Qwen3-ASR weights | model card: `apache-2.0`.[21] |
| Qwen3-TTS weights | model card: `apache-2.0`.[22] |
| Nemotron 3.5 ASR weights | model card: `openmdw-1.1`, ссылка на OpenMDW license.[23][24] |

Это только фиксация опубликованных upstream metadata. Настоящий отчёт **не делает юридического вывода** о допустимости конкретной дистрибуции, модификации, bundling или коммерческого использования.

## 8. Зрелость upstream

- Текущий `main`: `502b5b74...`, 2026-08-15 19:10:16 UTC.[1]
- Latest release: `release-0.6`, опубликован 2026-08-13.[2]
- Репозиторий создан 2026-06-23: проект молодой, но активный.
- После локального audit clone `9a5eb864...` upstream продвинулся на два commit: исправление Qwen language propagation и dropping out-of-span chunk word timestamps. Это показывает активную разработку, но также подвижность ASR contracts.[1]
- В clone учтено 288 tracked paths под `tests/`, включая 123 C++ и 79 Python files. Это inventory, не число исполняемых тестов.
- На exact head успешны шесть CI jobs: Linux CPU/Vulkan, Nix CPU/Vulkan, macOS CPU, Windows CPU.[17]
- **CUDA CI на head не найден**, поэтому RTX 3060 path остаётся локальным acceptance gate.
- На момент bounded API check: 6 open issues и 5 open PRs в первой полной странице; среди PR есть native UI, native package manager и WIP benchmark work.[18][19]

Итоговая характеристика: **active but young**, не заброшенный, но ещё не достаточно стабильный для замены всех incumbent runtimes без локального pinning и regression evidence.

## 9. Решения ACCEPT / MODIFY / DEFER / REJECT

| Предложение | Решение | Причина |
|---|---|---|
| Полностью заменить Python ASR/TTS и Faster-Whisper | **REJECT** | Whisper отсутствует; contracts и artifacts неэквивалентны |
| Добавить audio.cpp ASR provider сейчас | **DEFER** | нет идентичного 50-case runtime evidence |
| Изолированный standalone spike | **ACCEPT** | минимальный риск, сравнимый corpus |
| Optional ASR provider после gates | **MODIFY** | только batch сначала, family capabilities раздельно |
| Сразу включить streaming | **DEFER** | VOP contract пока finite-audio; нужен session lifecycle |
| Перевести Qwen TTS на audio.cpp | **DEFER** | offline-only runtime, другой artifact/voice contract |
| Удалить Faster-Whisper | **REJECT** | audio.cpp не предоставляет Whisper family |
| Оставить production без изменений | **ACCEPT** | сохраняет текущие acceptance lanes |

## 10. Рекомендуемая интеграционная архитектура

Минимальный blast radius:

```text
voiceover transcribe
  → cli.transcribe_cmd
  → ASRProviderRegistry
  → AudioCppASRProvider
  → subprocess audiocpp_cli (argv list, shell=False)
  → temporary text/words/metrics files
  → typed ASRResult + ASRExecutionReceipt
  → existing VOP JSON/exit-code owner
```

Предлагаемые owners:

- **новый:** `src/voiceover_pipeline/providers/audio_cpp_asr.py`;
- registration only: `providers/asr_registry.py`;
- dependency/config probing: существующий doctor/config owner;
- tests: `tests/test_audio_cpp_asr_provider.py`, registry/CLI/doctor tests;
- benchmark adapter после spike: `asr_benchmark.py`, `tools/run_asr_benchmark.py`;
- docs: `docs/agent-cli-contract.md`, provider guide, `docs/README.md`.

Первый provider:

- один ID, например `audio-cpp-local`;
- explicit allowlist families `qwen3_asr`, затем `nemotron_asr`;
- batch finite audio only;
- никаких downloads/conversion;
- executable и model package задаются явно;
- `subprocess` только argv list, `shell=False`, timeout/cancellation/cleanup;
- transcript читается из private temporary `--text-out`, а не из mixed stdout;
- `--words-out` запрашивается только при доказанной family capability;
- stdout/stderr не попадают в public JSON и проходят redaction;
- cloud и incumbent providers не изменяются.

## 11. Implementation plan после успешного spike

1. Зафиксировать spike verdict, audio.cpp SHA, binary/model hashes и accepted families.
2. Добавить provider module с typed subprocess runner и family allowlist.
3. Реализовать dependency probe: executable present, version/SHA known, selected loader listed, model package readable. **Не запускать inference в doctor.**
4. Зарегистрировать `ASRProviderSpec`; для первого slice объявить только реально доказанные capabilities.
5. Map Qwen context отдельно; для Nemotron разрешить только language locale. Reject glossary/phrase hints до отдельной реализации.
6. Преобразовать text/words в `ASRResult`, проверить monotonic timestamps и не выдавать default `0` confidence как реальную confidence.
7. Заполнить receipt: runtime, exact binary revision, model revision/digest, device, compute, wall/RTF/resource measurements.
8. Добавить offline mocked tests: listing, doctor, argv mapping, malformed/missing outputs, timeout, exit mapping, privacy, JSON stdout, cleanup.
9. Добавить opt-in installed test, требующий заранее установленный binary/model; не включать GPU/download в default suite.
10. Обновить machine contract и docs index.
11. Повторить full offline suite и exact 50-case acceptance.
12. Rollback остаётся удалением одного registry entry/provider module; incumbent paths не затрагиваются.

## 12. Проверяемый spike

### 12.1. Scope

- exact corpus: `tests/fixtures/wvm_slice5_benchmark/manifest.json`;
- 50 assets, те же SHA-256/reference/category;
- baselines: принятый Qwen report, текущие Whisper baseline receipts и только после acceptance — Nemotron report;
- отдельная директория, например `out/asr-benchmarks/audio-cpp-spike-<sha>/`;
- production source не изменяется;
- модели не скачиваются во время run.

### 12.2. Command shape

Single-case primitive:

```bash
/path/to/audiocpp_cli \
  --task asr \
  --family qwen3_asr \
  --model /pinned/models/Qwen3-ASR-0.6B-GGUF \
  --backend cuda \
  --device 0 \
  --mode offline \
  --audio /resolved/case.wav \
  --language ru \
  --text-out /private/tmp/<case>.txt \
  --metrics
```

Qwen context-on:

```bash
... --text "$(read approved prompt from non-.env file)"
```

Nemotron:

```bash
/path/to/audiocpp_cli \
  --task asr \
  --family nemotron_asr \
  --model /pinned/models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF \
  --backend cuda \
  --mode offline \
  --audio /resolved/case.wav \
  --language ru-RU \
  --text-out /private/tmp/<case>.txt \
  --metrics
```

Spike harness должен пройти manifest serially и записать:

- `runtime-receipt.json`;
- `resource-gate.json`;
- privacy-safe `asr-benchmark.json/.md`;
- per-case asset hash, WER/CER, wall time, RTF, error category;
- peak RAM/VRAM, cold/warm load;
- words/timestamp metrics только для действительно emitted output;
- stderr/stdout logs без reference text, transcripts и prompt contents.

### 12.3. Privacy/resource gates

До каждого model load и после cleanup использовать действующую локальную resource policy:

- три samples;
- reject при занятом WVM/Whisper GPU process;
- reject при sustained utilization выше текущего gate;
- reject при недостаточном free VRAM или temperature gate;
- stop on `nvidia-smi` error;
- serial execution;
- после run GPU/process state должен вернуться к baseline.

Не писать transcripts, reference text или prompt contents в report. Temporary outputs удалять после metric computation.

### 12.4. Go/no-go

**GO** только если одновременно:

1. все 50 cases завершены без crash/OOM/parser failures;
2. aggregate и critical subset WER/CER не хуже соответствующего incumbent больше, чем заранее зафиксированная повторная вариативность baseline;
3. prompt-on/off semantics сравнимы только там, где family действительно поддерживает context;
4. latency/RTF или memory дают заранее заявленную существенную пользу, оправдывающую второй runtime;
5. два cold/warm повторения с теми же hashes воспроизводимы;
6. JSON/exit mapping, privacy и cleanup проходят;
7. заявленные words/streaming capabilities доказаны output, а не флагами.

**NO-GO** при material quality regression, нестабильных outputs, OOM, утечке private text, отсутствии cleanup, mutable/unpinned package, недостоверных timestamps/confidence или отсутствии практического преимущества над incumbent. Абсолютный WER/CER threshold здесь намеренно не выдумывается: gate относительный к тем же assets и принятым baselines.

## 13. Риски, non-claims и rollback

### Риски

- молодой и быстро меняющийся upstream;
- отсутствие CUDA CI;
- spec/loader capability drift;
- mutable model package revisions;
- дополнительный C++/CUDA build и support burden;
- model-specific prompt semantics;
- generic outputs могут выглядеть поддержанными, оставаясь пустыми;
- transcript leakage через stdout/stderr;
- subprocess cancellation и orphaned GPU memory;
- duplicate model storage и build caches;
- Qwen forced aligner добавляет отдельные weights/runtime costs;
- OpenMDW license требует отдельной оценки для выбранной доставки.

### Non-claims

Отчёт не утверждает:

- что audio.cpp Qwen/Nemotron/TTS помещаются в 12 GB VRAM;
- что GGUF quality равна Python/Safetensors;
- что Nemotron benchmark VOP уже принят;
- что audio.cpp confidence meaningful;
- что streaming semantics готовы для VOP;
- что official model capability автоматически присутствует в audio.cpp;
- что framework и weights имеют одинаковую лицензию;
- что production provider уже реализован.

### Rollback

Incumbent остаётся default. Optional provider не меняет stored artifacts или registry IDs других providers. При regression:

1. отключить registration `audio-cpp-local`;
2. сохранить benchmark/receipt для анализа;
3. удалить только optional integration code/package instructions;
4. не менять Qwen/Nemotron Python, Faster-Whisper или cloud routes;
5. повторить generic CLI offline suite.

## 14. Решение для follow-up документационного коммита

1. **Включить этот research report в отдельный follow-up docs commit после уже опубликованного feature commit `f3cb5db77628221ef7750cd985b42158c574e1b9`.**
2. **Не включать production audio.cpp backend до завершения отдельного spike.**
3. Не менять default providers, artifacts, CLI machine contract или dependencies.
4. audio.cpp не должен блокировать текущие Qwen, Nemotron или Whisper acceptance gates.
5. Завершённый Nemotron benchmark относится к собственному ASR acceptance и не доказывает пригодность audio.cpp.
6. Следующее разрешённое действие по audio.cpp — только isolated standalone spike на pinned binary/models и тех же 50 assets.

## Sources

[1] https://github.com/0xShug0/audio.cpp/commit/502b5b74bd26e9b4aed267d1776ecf131cae7215 — audio.cpp commit `502b5b74bd26e9b4aed267d1776ecf131cae7215`
[2] https://github.com/0xShug0/audio.cpp/releases/tag/release-0.6 — audio.cpp release `release-0.6`
[3] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/README.md — audio.cpp README at exact commit
[4] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/model_specs_v1/qwen3_asr.json — Qwen3-ASR model spec
[5] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/qwen3_asr/loader.cpp — Qwen3-ASR loader
[6] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/qwen3_asr/session.cpp — Qwen3-ASR session
[7] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/model_specs_v1/nemotron_asr.json — Nemotron ASR model spec
[8] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/nemotron_asr/loader.cpp — Nemotron ASR loader
[9] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/nemotron_asr/session.cpp — Nemotron ASR session
[10] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/model_specs_v1/qwen3_tts.json — Qwen3-TTS model spec
[11] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/qwen3_tts/loader.cpp — Qwen3-TTS loader
[12] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/qwen3_tts/session.cpp — Qwen3-TTS session
[13] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/docs/usage.md — audio.cpp CLI usage
[14] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/app/cli/main.cpp — audio.cpp CLI implementation
[15] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/app/workflow/file_sink.cpp — audio.cpp result file sink
[16] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/docs/build/linux.md — audio.cpp Linux build guide
[17] https://github.com/0xShug0/audio.cpp/actions/runs/31903160054 — CI runs for exact head
[18] https://github.com/0xShug0/audio.cpp/issues — Open audio.cpp issues
[19] https://github.com/0xShug0/audio.cpp/pulls — Open audio.cpp pull requests
[20] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/LICENSE — audio.cpp LICENSE
[21] https://huggingface.co/Qwen/Qwen3-ASR-0.6B — Qwen3-ASR-0.6B model card
[22] https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice — Qwen3-TTS CustomVoice model card
[23] https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b — Nemotron 3.5 ASR Streaming 0.6B model card
[24] https://openmdw.ai/license/1-1 — OpenMDW License 1.1
[25] https://github.com/0xShug0/audio.cpp/issues/222 — audio.cpp issue #222: native model management
