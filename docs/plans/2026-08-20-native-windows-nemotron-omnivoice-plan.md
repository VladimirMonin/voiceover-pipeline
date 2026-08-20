# Native Windows plan: Nemotron prompts/timestamps and OmniVoice clone/design

> **Status:** in progress; offline foundation landed; native live acceptance pending.
> This document authorizes no download, tool installation,
> model acquisition, live GPU inference, publication, commit, or push. Each such
> action requires the owner's explicit approval when its gate is reached.

## Goal

Provide two fully native Windows routes without Docker, WSL, Wine, or a cloud
fallback:

- `nemotron-local` performs CUDA ASR, accepts a typed contextual prompt, and
  returns validated native word timestamps.
- `omnivoice-local` performs CUDA TTS in `clone` and `design` modes through a
  native `audiocpp_cli.exe` package.

The public provider IDs, JSON stdout contract, resumable artifact behavior, and
Faster-Whisper route remain intact. The target is not a broad replacement of
other local or cloud providers.

## Fixed Boundaries

- Native Windows only. Docker, WSL, Wine, a Linux container, and an implicit
  Python fallback are not acceptable implementations of these two routes.
- Models, build outputs, caches, temporary audio, and receipts live outside the
  Git checkout. The application never automatically downloads or converts model
  weights.
- One GPU-heavy model process runs at a time. Whisper Voice Machine retains
  priority when it owns the GPU.
- A native package must include its executable, non-system DLL closure, model
  artifacts, and SHA-256 manifest. A bare `.exe` or an arbitrary GGUF is not an
  accepted install.
- Prompting is a real feature requirement for Nemotron. Passing a language locale
  alone does not satisfy it, and no prompt capability may be claimed until a
  native live run proves the selected prompt contract.
- OmniVoice remains subject to its admitted artifact provenance and
  non-commercial license boundary. Clone and design do not relax that boundary.
- Offline automated tests use fixtures, fakes, and temporary directories only.
  They must not load a model, require CUDA, access the network, or read `.env`.

## Target Installation Shape

The exact filenames for the Nemotron package are determined during the native
runtime compatibility spike. The package root is fixed conceptually and stays
outside the repository:

```text
<local-model-root>\audio-cpp-<pinned-revision>\
├── audiocpp_cli.exe
├── audio_cpp_dependency_closure.json
├── build_receipt.json
├── <required native DLL files>
└── models\
    ├── nemotron\
    │   └── <verified native Nemotron artifact package>
    └── omnivoice\
        └── omnivoice-q8_0.gguf
```

The implementation may provide one non-secret root setting and explicit
per-artifact overrides. It must not put absolute model paths into public JSON
receipts or logs.

## Current Seams

The implementation is constrained by the existing code rather than a rewrite:

- `src/voiceover_pipeline/local_runtime/gpu_lease.py` currently depends on the
  POSIX-only `fcntl` module.
- `src/voiceover_pipeline/local_runtime/lifecycle.py` has Linux-specific process
  inspection through `/proc`.
- `src/voiceover_pipeline/local_runtime/transports/audio_cpp_cli.py` already has
  a Windows process-group launcher, Unicode-safe argument transport, private
  workspaces, and dependency-closure validation.
- `src/voiceover_pipeline/providers/audio_cpp_nemotron_asr.py` explicitly does
  not select the native route on Windows and currently rejects all prompt hints.
- `src/voiceover_pipeline/providers/audio_cpp_omnivoice_tts.py` selects a native
  package on Windows, but supports only the fixed female-style mode.
- `src/voiceover_pipeline/cli.py` exposes no ASR context/prompt option for
  `transcribe`; `ASRContextHints` already has typed fields in
  `src/voiceover_pipeline/models.py`.
- `src/voiceover_pipeline/local_runtime/contracts.py` currently models only Qwen
  TTS modes, so OmniVoice clone/design needs a family-safe contract extension.

The broader architecture and existing safety constraints are documented in the
[hybrid runtime plan](2026-08-16-audio-cpp-hybrid-migration-plan.md), the
[runtime contract](../audio-cpp-runtime.md), and the
[research addendum](../research/2026-08-16-audio-cpp-hybrid-consolidation.md).

## Definition Of Done

The work is complete only when all of the following are true on a native Windows
host with an NVIDIA GPU:

- `voiceover doctor --json` imports and runs without POSIX-only failures.
- A native Nemotron invocation accepts a documented contextual prompt and
  returns text plus monotonic, transcript-matching word timestamps.
- A native OmniVoice clone invocation requires reference audio and its exact
  transcript, while a design invocation accepts an explicit design instruction.
- Every native invocation returns one JSON object on stdout under `--json` and
  does not disclose private paths, prompt text, or reference audio text in public
  receipts.
- Native package, model identity, build identity, GPU cleanup, cancellation, and
  repeated-run stability have live evidence.
- No Docker or WSL executable is launched or documented as a fallback.

## Work Series

Every code-producing task starts with an offline failing test, implements the
smallest coherent change, reruns the focused test, and then runs the required
lint, format, and type gates on the exact bytes. A Kanban card for each task must
record Codebase, Serena, and ast-grep evidence before implementation.

### Делаем раз: зафиксировать native Windows контракт

Задачи:

- Create a Windows acceptance matrix covering import, package admission, model
  identity, CUDA preflight, prompt mapping, word timestamp validation, clone,
  design, cancellation, cleanup, and JSON error behavior.
- Define one explicit Nemotron prompt contract. The minimum public capability is
  typed contextual text; the design must state whether `initial_prompt`, glossary,
  and phrase hints are separately supported or rejected.
- Add target CLI arguments for the supported contract, for example a mutually
  exclusive `--context` or `--context-file`. Do not add arguments whose native
  semantics cannot be proved.
- Define a runtime-selection argument for ASR so callers can require native
  Windows execution and receive a fail-closed error instead of an implicit Python
  fallback.
- Define family-safe OmniVoice request modes rather than reusing Qwen-only mode
  names without validation. Clone must require both reference audio and reference
  transcript; design must require a non-empty instruction.
- Add deterministic contract and parser tests before provider changes.

Самопроверка:

- New tests prove unsupported prompt fields and invalid clone/design combinations
  fail before a model load.
- Existing CLI JSON and exit-code tests remain unchanged unless an approved public
  contract extension requires new assertions.
- The target prompt surface is documented as planned until the native compatibility
  spike proves it.

### Делаем два: сделать runtime импортируемым на Windows

Задачи:

- Replace the unconditional `fcntl` dependency with a cross-platform file-lock
  implementation. Keep equivalent cross-process ownership semantics on Windows.
- Add a Windows-safe process-liveness implementation and Windows-safe WVM process
  detection. If WVM ownership cannot be determined safely, fail closed instead of
  silently running a competing GPU job.
- Make optional native runtime imports lazy enough that a missing local model,
  native binary, or CUDA toolchain cannot prevent baseline CLI commands from
  importing.
- Preserve POSIX behavior and process-group cancellation on non-Windows hosts.
- Add unit tests for Windows and POSIX branches using mocks; do not depend on the
  host operating system in the test result.

Самопроверка:

- `python -B -c "import voiceover_pipeline.cli"` succeeds on Windows.
- Unit tests cover lease acquisition, release, stale-owner recovery, cancellation,
  and no-`/proc` Windows behavior.
- `voiceover doctor --json` reports unavailable optional native components as
  structured health failures rather than an import traceback.

### Делаем три: подготовить native build и package contract

Задачи:

- Confirm the pinned `audio.cpp` source revision, supported Windows CUDA build
  flags, model-family flags, and exact `audiocpp_cli` syntax from the checked-out
  upstream source before changing VOP integration.
- Install or provision the required native toolchain only after explicit approval:
  MSVC, CMake/Ninja or Visual Studio generator, CUDA Toolkit, and a compatible
  NVIDIA driver.
- Build `audiocpp_cli.exe` outside both the VOP checkout and upstream source tree.
- Generate a reproducible build receipt containing source revision, compiler,
  CMake, CUDA toolkit, architecture, build flags, binary SHA-256, and included
  model families.
- Generate `audio_cpp_dependency_closure.json` from the actual executable and its
  non-system DLL dependencies.
- Extend package admission so it can validate an admitted model directory as well
  as a single GGUF file, without reading all large files into memory at once.
- Add offline tests for malformed manifests, missing DLLs, model paths outside the
  package, modified model bytes, and Unicode/space-containing Windows paths.

Самопроверка:

- Package verification accepts only a complete, checksummed package.
- A direct native `audiocpp_cli.exe --help` and CUDA backend probe are performed
  only after explicit approval and are recorded as local evidence, not as a unit
  test.
- No model artifact, build directory, cache, or binary appears in `git status`.

### Делаем четыре: подключить native Nemotron с prompt и timestamps

Задачи:

- Implement the Windows branch of `AudioCppNemotronASRProvider` using the existing
  `AudioCppNativeCLITransport`; remove the current Windows rejection only after
  package admission succeeds.
- Stage arbitrary supported input audio into a private 16 kHz mono PCM WAV before
  the native process starts.
- Map the accepted typed contextual prompt into the exact native Nemotron request
  mechanism established by the compatibility spike. Never map it to a language
  field, silently discard it, or invent prompt semantics.
- If the pinned upstream CLI lacks the required prompt mechanism, implement and
  pin a minimal native extension, then test its request/response contract directly.
- Parse transcript, segments when available, and native timestamp output. Normalize
  RNN-T token entries into transcript-matching word spans.
- Validate non-negative, monotonic, non-overlapping timestamps within the audio
  duration. Preserve `alignment_origin="native"` only when native words passed
  validation.
- Make native runtime selection fail closed when the package, CUDA state, prompt
  support, or timestamp output is unavailable.
- Add mocked provider, transport, CLI, long-form offset, error-mapping, and
  regression tests for both prompt-on and prompt-off requests.

Самопроверка:

- Focused tests prove that prompt text reaches the native command/request only for
  an explicitly supported field.
- Focused tests prove word timestamps are rejected when missing, reversed,
  out-of-bounds, non-monotonic, or incompatible with the returned transcript.
- The Python Nemotron route retains its truthful text-only behavior and cannot be
  mislabeled as native timestamp support.
- `--json` emits exactly one object; diagnostics stay on stderr.

### Делаем пять: доказать Nemotron на реальной Windows GPU

Задачи:

- Acquire the approved native Nemotron artifact and record source revision,
  license/provenance metadata, file hashes, and package location outside Git.
- Run a direct, finite-audio native CLI smoke test before exercising VOP.
- Run VOP with prompt off, then with the same audio and an approved contextual
  prompt. Keep audio, expected terms, prompt digest, transcript, and timestamps
  in a local non-secret evidence directory.
- Run a short Russian word-timestamp fixture, a long-form chunk-offset fixture,
  a no-speech fixture, and a cancellation/timeout fixture serially.
- Record duration, wall time, VRAM/RAM peak, GPU cleanup, process cleanup, and
  timestamp invariant outcomes.
- Repeat the successful prompt-on path enough times to expose cold/warm and
  stability failures; never run two GPU model processes in parallel.

Самопроверка:

- The prompt-on evidence shows an intentional, inspectable result difference or
  an upstream-defined acknowledgement. Mere command-line presence is insufficient.
- Every speech word span maps to the transcript, remains within source duration,
  and has `alignment_origin="native"`.
- The process exits, the GPU lease is released, and no private prompt/reference
  text is exposed by public artifacts.
- A failure in any live gate leaves the provider unpromoted and preserves the
  existing Python route as the rollback path.

### Делаем шесть: расширить OmniVoice до clone и design

Задачи:

- Extend `LocalTTSRequest`, the native CLI codec, and `OmniVoiceLocalTTSProvider`
  with OmniVoice-specific fixed-style, clone, and design modes.
- Update CLI validation so `omnivoice-local --mode clone` accepts only a readable
  reference audio file plus non-empty reference transcript.
- Update CLI validation so `omnivoice-local --mode design` accepts only a non-empty
  style/design instruction. The existing fixed-style mode remains explicit.
- Confirm the exact pinned native CLI tasks and arguments through the built binary.
  Do not assume Qwen task names apply to OmniVoice.
- Stage reference audio in a private workspace, validate its duration and format,
  and pass only the staged path to the child process.
- Keep long-form sentence packing and choose a session strategy separately for
  fixed-style, clone, and design. Do not silently collapse separately requested
  voice identities into one session.
- Extend public metadata to describe mode and admitted artifact provenance without
  raw local paths, reference transcript, or style prompt text.
- Add deterministic tests for all accepted and rejected combinations, WAV output
  validation, staging cleanup, Unicode paths, cancellation, and receipt redaction.

Самопроверка:

- Clone without a transcript fails before native launch.
- Design without an instruction fails before native launch.
- Existing fixed female-style behavior remains covered and does not regress.
- No Qwen-only option is accidentally accepted as an OmniVoice clone/design
  option unless the native OmniVoice contract explicitly supports it.

### Делаем семь: доказать OmniVoice clone и design на реальной Windows GPU

Задачи:

- Complete a separate license and provenance review of the admitted OmniVoice
  artifact before acquisition or live use.
- Verify the exact artifact SHA-256 before every provider admission.
- Run direct native clone and design smoke tests with owner-approved, non-private
  or explicitly permitted reference audio.
- Run VOP clone and design paths on short Russian scripts, then on a bounded
  long-form script with sentence packing.
- Measure output WAV validity, 24 kHz mono properties, duration, cold/warm wall
  time, VRAM/RAM peak, output continuity, cancellation, process cleanup, and
  release of the GPU lease.
- Repeat each successful mode serially to check stability and reference isolation.

Самопроверка:

- Clone output is tied to the supplied reference workflow but neither reference
  audio nor transcript is copied into public receipts.
- Design output receives the approved instruction through the native runtime; the
  instruction itself remains private.
- Output is a readable non-empty mono WAV at the expected sample rate before VOP
  converts it to MP3.
- Any license, hash, quality, resource, or cleanup failure leaves OmniVoice
  unpromoted.

### Делаем восемь: закрепить CLI, documentation, and release gate

Задачи:

- Update [Agent CLI Contract](../agent-cli-contract.md) only after the final flags,
  capability states, exit codes, and JSON fields are proven by code and tests.
- Update Windows-facing local model documentation with non-secret setup examples,
  explicit package layout, model admission rules, and the absence of Docker/WSL
  fallback.
- Document the model provisioning workflow separately from ordinary application
  installation. It must require an explicit approval for network download or
  model acquisition.
- Add regression coverage for listing, doctor, provider selection, native package
  health, JSON output, resume behavior, and no private-path leakage.
- Run the narrow test groups first, then all relevant tests. For each Python code
  change run `uv run ruff check src tests`,
  `uv run ruff format --check src tests`, and
  `uv run mypy --no-incremental`.
- Perform an independent review against this plan before any promotion, commit,
  tag, push, or release decision.

Самопроверка:

- Documentation does not claim a native Windows model works until the corresponding
  live gate has passed.
- `doctor` distinguishes missing tooling, invalid native package, missing admitted
  model, unavailable CUDA, unsupported prompt contract, and failed runtime health.
- `git diff --check` passes, tests remain offline/deterministic, and no generated
  assets or secret-bearing configuration files enter the change set.
- No commit, tag, push, model redistribution, or publication occurs without a
  separate explicit owner approval.

## Final Acceptance Record

Before marking the series complete, record locally:

- Windows version, NVIDIA driver, GPU name, CUDA toolkit, MSVC, CMake, and
  `audio.cpp` source revision.
- SHA-256 closure for the executable, DLLs, Nemotron package, and admitted
  OmniVoice artifact.
- Exact runtime choice, model IDs, prompt capability state, and timestamp origin.
- Focused automated-test commands and outcomes.
- Live run inputs by safe identifier, output artifact hashes, timing/resource
  measurements, cleanup outcome, and any known limitations.

Do not place private audio, prompt text, reference transcripts, absolute user
paths, model weights, or credentials in this record or in Git.

## Closing Voiceover

After the final acceptance record is written, generate a closing voiceover
narrating the completed work in plain, non-technical language. Use the female
voice (OmniVoice, `omnivoice-local` fixed-style preset on the native Windows
route). The script must summarize what was implemented, what was verified, and
what remains as known limitations — without absolute paths, model weights,
prompt/reference text, credentials, or anything private.

## Execution Ledger — зафиксирован целиком 2026-08-20

Берём максимальную цель: полный native Windows DoD, включая Nemotron, OmniVoice clone/design, live-приёмку, документацию и финальный женский аудиоотчёт.

**Текущее состояние**
Пять сабов независимо подтвердили:

- Серии 1–2 завершены.
- Серия 3 завершена только offline: admission, receipts и тесты есть, реальной Windows-сборки ещё нет.
- Nemotron на Windows по-прежнему заблокирован в `from_environment`, контекст отклоняется, CLI не разрешает `--runtime audio-cpp`.
- OmniVoice clone/design существуют только как контракт и CLI-валидация; provider и transport поддерживают только fixed-style.
- Текущий package admission не связывает receipt с pinned revision, CUDA и требуемыми model families.
- Codebase/Serena недоступны; ast-grep подтвердил основные точки интеграции. По правилам репозитория восстановление code-intelligence routes входит в первый gate.

**Документ-план**
Не создаём второй конкурирующий план. Расширяем:

`docs/plans/2026-08-20-native-windows-nemotron-omnivoice-plan.md`

Первое изменение документа:

- Статус: `in progress; offline foundation landed; native live acceptance pending`.
- Таблица состояния всех серий.
- Execution ledger с карточками ниже.
- Decision log.
- Approval/live-gate register.
- Шаблоны evidence и rollback.
- Ссылки на коммиты `a845a7c`, `dee8854`, `64df7ea`.
- Исторический `Current Seams` помечается состоянием на `2f420c2`.

## Волна 0: Preflight

| Карточка | Работа | Критерий завершения |
|---|---|---|
| `NW-00` | Синхронизировать план и `docs/README.md` | Документы отражают фактический статус без live-заявлений |
| `NW-01` | Восстановить Codebase/Serena, собрать ast-grep evidence | Cross-layer implementation разрешён правилами репозитория |
| `NW-02` | Автоматически обнаружить local upstream, MSVC, CMake, CUDA, native package и модели | Получена безопасная матрица `present/missing/version`, без чтения `.env` и вывода приватных путей |

Preflight не устанавливает и не запускает модели.

## Волна 1: Заморозить upstream-контракт

### `NW-03`: audio.cpp compatibility spike

Из pinned source и реального `audiocpp_cli.exe --help` фиксируем:

- Windows CMake/CUDA flags и target name.
- Nemotron family token, model shape и output schema.
- Настоящий механизм contextual prompt.
- Единицы и timebase word timestamps.
- OmniVoice fixed/clone/design tasks и аргументы.
- Требования к reference audio и transcript.
- Возможность file/stdin transport для приватного текста.

Если pinned revision не поддерживает обязательный prompt или clone/design:

- Делаем минимальное native-расширение.
- Коммитим его в отдельном upstream checkout.
- Задаём новую immutable revision.
- Не выдаём patched binary за исходный `502b5b...`.

Никакие Qwen task names не переиспользуются по предположению.

## Волна 2: Реальный native package

### `NW-04A`: Package hardening

Изменения:

- Receipt обязан совпадать с pinned source revision.
- Receipt обязан объявлять `cuda`, архитектуру и нужные model families/features.
- Все build identity fields становятся обязательными.
- Admission возвращает проверенные receipt facts.
- Legacy `discover_native_audio_cpp_install` делегирует единому admission.
- Transport принимает как model file, так и directory package.
- Исправляется ложное распознавание MSVC-флагов вроде `/O2` как absolute paths.
- Receipt и closure записываются атомарно.
- Source checkout проверяется на pinned HEAD и чистоту.

Основные файлы:

- `scripts/build_audio_cpp.py`
- `audio_cpp/inventory.py`
- `audio_cpp_package.py`
- `audio_cpp_cli.py`

### `NW-04B`: Build и closure

- Out-of-tree Windows build.
- Рекурсивный PE dependency scan.
- Исключение системных DLL.
- Staging immutable package вне Git.
- Генерация receipt и closure по фактическим байтам.
- Independent hash verification.
- Central package admission.
- `--help` и CUDA probe только после structural admission.

## Волна 3: Общий native transport

### `NW-05`: Private staging и codec

Один координатор владеет общим `audio_cpp_cli.py`, чтобы сабы не конфликтовали.

Реализуем:

- ASR input всегда staging в 16 kHz mono PCM WAV.
- Duration измеряется по staged WAV.
- Reference audio копируется под нейтральным именем в private workspace.
- Prompt, transcript и design instruction передаются через доказанный file/stdin механизм.
- Sensitive text не появляется в argv, exception, stdout, receipt или metadata.
- Cleanup работает после success, error, timeout и cancellation.
- Unicode/space paths покрываются тестами.
- OmniVoice output строго проверяется как non-empty 24 kHz mono WAV.
- Nemotron raw entries сохраняют `keep`, offsets и upstream timing fields.

## Волна 4: Offline implementation

### `NW-06`: Nemotron

- Windows provider создаётся только после package admission.
- `--runtime audio-cpp` выбирает native route и никогда не откатывается на Python.
- Native route требует CUDA.
- `context_text` отображается только в доказанное upstream-поле.
- Glossary, phrase hints и initial prompt остаются fail-closed.
- Native words нормализуются и проверяются на monotonicity, bounds и transcript correspondence.
- `alignment_origin="native"` выставляется только после полной валидации.
- Long-form offsets применяются ровно один раз.
- Execution receipt различает runtime revision и model revision.
- Doctor возвращает точные reason codes.
- Python Nemotron остаётся явным rollback route.
- `auto` пока не продвигается.

### `NW-07`: OmniVoice

- `OmniVoiceRequest` подключается к `LocalTTSRequest` как family-specific contract.
- Sensitive fields получают `repr=False`.
- Provider принимает fixed-style, clone и design.
- CLI перестаёт возвращать `not implemented` для валидных запросов.
- Clone требует reference audio и точный transcript.
- Design требует instruction и запрещает reference fields.
- Session strategy учитывает mode и identity.
- Разные references/instructions не объединяются в одну сессию.
- Public metadata содержит mode и provenance, но не paths/text.
- Windows metadata больше не сообщает `single-container`.
- Capability listing остаётся консервативным до live acceptance.

Обе карточки получают отдельного implementer-саба и отдельного reviewer-саба.

## Волна 5: Offline acceptance

### `NW-08`

Фокусные тесты:

```powershell
uv run pytest -q `
  tests/test_audio_cpp_package.py `
  tests/test_audio_cpp_build_producer.py `
  tests/test_audio_cpp_native_cli.py `
  tests/test_audio_cpp_nemotron_asr.py `
  tests/test_nemotron_asr_provider.py `
  tests/test_nemotron_word_normalization.py `
  tests/test_asr_contract.py `
  tests/test_asr_cli.py `
  tests/test_asr_longform.py `
  tests/test_audio_cpp_contracts.py `
  tests/test_audio_cpp_omnivoice_tts.py `
  tests/test_audio_cpp_omnivoice_cli.py `
  tests/test_cli_validation.py `
  tests/test_cli_json_contract.py `
  tests/test_generation_stability.py `
  tests/test_local_tts_text.py
```

Обязательные gates:

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy --no-incremental
uv run pytest
git diff --check
```

Полный baseline сравнивается в отдельном clean worktree, без stash/reset текущего дерева. Новые failures в package, native Windows, privacy, JSON или lifecycle не допускаются.

## Волна 6: Live acceptance

GPU-тесты идут строго последовательно.

### `NW-09`: Nemotron live

- Direct finite-audio smoke.
- VOP prompt-off.
- Тот же audio с prompt-on.
- Подтверждение prompt effect или native acknowledgement.
- Короткий русский timestamp case.
- Long-form case через границу chunk.
- No-speech case.
- Cancellation и timeout.
- Минимум три последовательных prompt-on run.
- Контроль VRAM/RAM, процессов, lease и cleanup.

### `NW-10`: OmniVoice live

- Fixed-style, clone и design direct smoke.
- Короткий русский VOP run для каждого режима.
- Clone A/B/A для проверки reference isolation.
- Design A/B/A для проверки instruction isolation.
- Bounded long-form clone/design.
- Cancellation во время clone staging/inference.
- Минимум три последовательных run каждого режима.
- Проверка WAV, MP3 conversion, quality, continuity, VRAM/RAM и cleanup.

Любая ошибка оставляет соответствующую capability непродвинутой.

## Волна 7: Документация и завершение

### `NW-11`: Final docs

Обновляются:

- `docs/agent-cli-contract.md`
- `docs/audio-cpp-runtime.md`
- `docs/omnivoice-local-tts.md`
- `docs/README.md`
- skill docs по providers/commands/troubleshooting

Создаются после фактической проверки:

- `docs/native-windows-audio-cpp-provisioning.md`
- `docs/nemotron-local-asr.md`
- `docs/reports/2026-08-20-native-windows-nemotron-omnivoice-acceptance.md`

Acceptance report содержит только safe IDs, versions, hashes, durations, timings и invariant booleans. Raw audio, prompts, transcripts, paths и model weights остаются вне Git.

### `NW-12`: Closing voiceover

Только после sealed acceptance record:

- Native Windows `omnivoice-local`.
- Fixed-style female preset.
- Простой русский рассказ о реализации, проверках и ограничениях.
- MP3, script, hash и metadata вне Git.
- Проверка duration, audio format, cleanup и отсутствия приватных данных.

## Коммиты

Планируем отдельные коммиты:

1. `docs: synchronize native Windows execution plan`
2. `feat: harden native audio.cpp package identity`
3. `feat: add private native audio staging`
4. `feat: enable native Windows Nemotron ASR`
5. `feat: add OmniVoice clone and design modes`
6. `docs: record native Windows acceptance`

Каждый коммит создаётся только после focused tests и independent review. Push выполняется после проверки всей включённой серии.

## Реалистичный срок

При наличии upstream, toolchain и моделей:

| Этап | Оценка |
|---|---:|
| Plan/preflight | 30–60 минут |
| Compatibility spike | 45–90 минут |
| Package/build | 1.5–3 часа |
| Offline implementation | 3–5 часов |
| Offline review | около 1 часа |
| Live acceptance | 2–4 часа |
| Docs/voiceover | 45–90 минут |

Полный путь занимает примерно 8–14 часов. Если upstream не имеет prompt или clone/design и потребуется C++ extension, это главный риск выхода за сегодня. В таком случае offline-кандидат всё равно доводится до зелёного состояния, но live capability не объявляется готовой.

После выхода из Plan Mode начинаем с `NW-00`, `NW-01` и автоматического preflight, без дополнительных открытых опросов.
