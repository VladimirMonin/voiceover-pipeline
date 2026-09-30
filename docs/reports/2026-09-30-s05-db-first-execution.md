# S05 — DB-first исполнение и восстановление

| Поле | Факт |
|---|---|
| Дата / ветка | 2026-09-30 / `main` |
| Кодовый HEAD | `01bca32f8f12ded6a02958564f223396a88006d0` — последний implementation commit; матрица, итоговый P1 fix и отчёт входят в отдельный scoped closure-commit |
| База S04 | принятый отчёт S04 `docs/reports/2026-09-29-s04-local-history.md` (`025ff62bce73b6747bfd76ec644dab4a0038885d`); docs-commit отчёта — `65b99e9` |
| Статус | **`ACCEPT_OFFLINE` (родитель после исправления P1, обнаруженного независимым Sol6)**; live/provider/listening/installed-model **NOT_RUN** |
| Граница Git | 28 локальных S05 implementation-коммитов по `01bca32`; последний ранее разрешённый push — `c0c56b6` в `origin/main`. Матрица/финальный P1 fix/этот отчёт входят в отдельный локальный scoped closure-commit без push; следующий push, tag и release не разрешены |
| Внешние инструменты | Codebase/Serena/ast-grep в этой дочерней сессии не предоставлены; использовано целевое чтение исходников, инструментальный пробел зафиксирован |

Отчёт фиксирует S05 на кодовом HEAD плюс scoped closure-diff. Независимый Sol6
(`1fcd677b-7013-420f-ab23-6bf1523130b8`) сначала выдал **BLOCK** на старых
байтах: `history resume` мог пройти через заменённый symlink-предок output root.
Родитель исправил именно этот P1 в том же S05 closure, получил fail-first тест
(до исправления exit `0` вместо безопасного отказа) и повторил финальные гейты.
**Sol6 не пересматривал исправленные байты**; итоговый `ACCEPT_OFFLINE` — решение
родителя по проверенному исправлению, а не приписанный ревьюеру вердикт.

## Область S05: 28 коммитов

`89d9fcc..01bca32` (без учёта docs-commit отчёта S04 `65b99e9`):

| Группа | Коммиты |
|---|---|
| Локи, guard'ы миграций | `89d9fcc`, `d1fa428` |
| Snapshot-идентичность и paid-маркер | `ac2eb39`, `d9d6ecc`, `555703f`, `df74f96`, `ce5cf29`, `7f782ca`, `d06fd21`, `118cbed` |
| Полза Media/синхронный TTS из истории | `c0c56b6`, `a5f0586` |
| `history resume/sync`, расходы | `f1c4da4`, `5fabffb`, `601fc7b` |
| Локальные ASR/timings/verify | `964d8ee`, `418b2fa`, `7aa4db0` |
| Диалоговые маршруты | `c7092c1`, `4420cdc` |
| Прочие локальные режимы и voiceover | `fd58778`, `b4364fe`, `2f48e95`, `a57ba8e`, `9b6d9ff` |
| Интегрированные шаги и облачный paid boundary | `30f3108`, `ad2c0cb`, `01bca32` |

## Матрица исполнения (finite, не Cartesian)

Диспетчер маршрута — `cli._native_route_eligible` (решение принимается до legacy
JSON recovery, чтения ключа, цен и удаления). «Native» = canonical SQLite-writer;
«legacy» = существующий JSON-writer без изменений.

| Маршрут | Формат/режим | Диспозиция |
|---|---|---|
| `polza-tts` `/media` (async) | markdown, voiceover | native |
| `polza-tts` `/audio/speech` (sync) | markdown, voiceover | native |
| `polza-chat-audio` | markdown, voiceover | native |
| `openrouter-tts` (обычный) | markdown, voiceover | native |
| `qwen-local` preset/clone/design | markdown, voiceover | native |
| `omnivoice-local` preset bank | markdown, voiceover | native |
| `omnivoice-local` auto/clone/design | markdown | native |
| `--no-trim` | любой нативный маршрут | native (записанная семантика) |
| `--with-timings` local `faster-whisper` | любой нативный маршрут | native |
| `--with-timings` paid `groq-whisper`/`xai-stt` | любой нативный маршрут | native (paid boundary) |
| локальный `--tts-quality-provider` `qwen-local`/`nemotron-local` | нативный не-диалоговый/диалоговый | native |
| `--tts-quality-provider xai-stt` | два диалоговых маршрута (per-turn) | native (paid boundary) |
| `openrouter-tts` Gemini dialogue (2 голоса) | dialogue | native (обязательный gate) |
| `omnivoice-local` preset bank dialogue | dialogue | native (опциональный gate) |
| локальные `transcribe` + `timings --asr-provider`/`faster-whisper` + `verify-tts` | stand-alone | native |
| stand-alone `timings --timing-provider groq-whisper\|xai-stt` | stand-alone | native (paid boundary) |
| `history resume ID` / `history sync ID` | по снимку | native DB-first |

Фактически неработающие/явно отклоняемые (не scope-cut):

| Маршрут | Диспозиция |
|---|---|
| `qwen-local --mode auto` | usage error (exit `2`) до провайдера/модели/снимка; режим не подменяется |
| `timings`/`--with-timings` `openrouter-whisper` | явный отказ: не даёт реальных таймстемпов; native-маршрут не выбран |
| `polza-tts`/`polza-chat-audio` dialogue | usage error: модель — не Gemini cast route |
| `omnivoice-local` non-preset + voiceover | usage error: режим отклоняет `--voice` из сценария |
| облачный/незарегистрированный quality на **не-диалоговом** прогоне | флаг игнорируется legacy executor'ом (сохранённое поведение) |
| будущий облачный ASR | вне S05 (S07), ничего не сохраняет |

## Что изменено в этой closure-задаче

Рабочее дерево (без commit):

- `tests/test_history_native_route_matrix.py` — новый consolidated
  parametrized dispatch-регресс: одна таблица `provider/mode/format/options →
  native|legacy`, плюс проверки `openrouter-whisper` (никогда не native) и
  `qwen-local --mode auto` (usage error без подмены режима). Полностью offline,
  без провайдера, ключа и файловых записей.
- `src/voiceover_pipeline/cli.py`, `src/voiceover_pipeline/history/native_asr.py`,
  `src/voiceover_pipeline/history/native_snapshot.py` — реконсиляция только
  фактов в docstring'ах: диалоговые маршруты больше не описаны как legacy для
  облачных timing/QA; «temporary S05 integration gap» и «single-writer wiring не
  существует» заменены на фактическое состояние; облачный timing описан как
  paid boundary, а не как «не wired here».
- `docs/agent-cli-contract.md`, `docs/skills/voiceover-pipeline/docs/06-commands-and-flags.md`,
  `docs/skills/voiceover-pipeline/docs/04-input-format.md`,
  `docs/skills/voiceover-pipeline/docs/14-local-audio-cpp-models.md` — сняты
  метки «фрагмент S05»/«S05 не завершён»/«standalone verify-tts на legacy»;
  облачный timing и облачный `xai-stt` dialogue QA описаны как переведённые на
  canonical/payed boundary, а не как legacy.
- `docs/README.md` — принятый родителем офлайн-статус S05 и ссылка на этот отчёт.
- После Sol6 BLOCK: `history resume` сверяет сохранённый output locator с
  committed `runs.run_root` **до и внутри root lock**, не создаёт каталог при
  replay; raw reader, crash-window recovery и atomic artifact writer не следуют
  symlink-предку. `tests/test_paid_transcription_timing.py` проверяет перенос
  родительского каталога, symlink на него, сохранённый raw и отказ без повторного
  POST/публикации. Основное поведение S05 не менялось.

Существующие legacy-корни и native-следы намеренно не менялись; новый функционал
не добавлялся.

## Доказанные офлайн-контракты

- **Один writer и владение.** Run-root lock один на native и legacy хвост;
  native-след без своей БД/строки — fail-closed (`NATIVE_OWNERSHIP_*`), не тихий
  JSON-fallback. `generate` отклоняет paid-owned root, а `timings` — native-owned.
- **Paid-маркер до POST.** Маркер попытки (`submitting`/`remote_accepted`)
  коммитится до запроса; неопределённый исход блокирует retry/fallback/resume/
  overwrite без второго POST; известный Polza Media ID доводится только GET.
- **Provenance/cost.** Родительский TTS и child `.paid-timing`/`.paid-quality`
  не смешиваются: parent raw/cost/audio нетронуты, child cost = unknown (NULL), не
  выдуманный ноль. Локальные `local_tts_chunk` считаются
  `local_attempts_without_api_charge`.
- **Privacy.** Приватные `tts_script`/`verification_transcript`/`asr_transcript`
  не печатаются в `--json`/`list`/`show`/`costs`; `verify-tts` никогда не хранит
  ожидаемый текст и не перезаписывает сценарий.
- **Resume/sync.** Восстановление из снимка без исходного `script.md`, тот же
  lock/CAS; `sync` не делает новый платный submit и не запускает локальные
  модели; `--overwrite` не принимается.

## Фактические проверки

На кодовом срезе и в рабочем дереве:

- `uv run --offline --frozen pytest -q -p no:cacheprovider` —
  **2052 passed, 2 skipped** на финальных байтах (skips: native Windows file
  locking на macOS и WVM reference validation без `WVM_ROOT`).
- `uv run --offline --frozen pytest -q -p no:cacheprovider
  tests/test_paid_transcription_timing.py tests/test_history_native_paid_processing.py
  tests/test_history_native_route_matrix.py tests/test_history_native_timing.py
  tests/test_history_native_resume.py` — **164 passed** после P1 fix.
- Fail-first `test_resume_refuses_retargeted_output_ancestor_before_replay` на
  старых байтах: **FAILED**, `assert 0 == 50`; после исправления — PASS,
  no POST, raw_saved сохранён, чужая директория не изменена.
- `uv run --offline --frozen ruff check src tests` — PASS;
  `ruff format --check src tests` — **160 файлов**; `mypy --no-incremental` —
  PASS, **84 source files**.
- `git diff --check` — PASS (пустой вывод, exit `0`).
- ast-grep/Codebase/Serena — **NOT_RUN_BY_THIS_SESSION** (инструменты не
  предоставлены дочерней сессии); маршрутность проверялась чтением
  `cli._native_route_eligible` и его helper'ов.

## Ручной fake/local interruption→resume smoke

Прогон выполнен на `01bca32` плюс первоначальный closure-diff **до** последней
починки symlink-предка; на финальных байтах маршрут восстановления повторно
проверен регрессионными тестами и полной offline-suite выше. Изолированный offline-прогон в временном `VOICEOVER_HOME` (POSIX `0700`), без
сети: monkeypatch `requests.sessions.Session.request` и `requests.get/post/...`
на tripwire, который бросает исключение на любом реальном запросе; ни `.env`, ни
ключ, ни загрузка модели, ни живой провайдер не использовались. Fake-провайдеры
инжектировались через `cli.build_provider`; FFmpeg/ASR-швы заменены in-memory.
Команда запуска: `uv run --offline --frozen python <smoke-script>`.

**Часть 1 — fake Polza `/media` accepted ID → history → GET-only resume.**

- Failpoint: fake `synthesize_chunk` вызвал `on_media_task_accepted("task-chunk_01")`
  и затем бросил `requests.Timeout` (симуляция обрыва после приёма).
- Наблюдение: `generate --json` → exit `30`, `error_code=NATIVE_SYNTHESIS_FAILED`;
  `submits=['chunk_01']`, `recovers=[]`; committed attempt
  `status=remote_accepted`, `remote_id=task-chunk_01`, `cost=null`.
- `history list` → 1 run `status=running`; `history show <uuid>` → попытка
  `remote_accepted` с `cost.amount=null`, `cost.source=unknown`, приватный
  transcript не выводится; `history costs` → `unknown_attempts=1`,
  `completeness=partial`, `local_attempts_without_api_charge=0`.
- `generate --resume` → exit `0`, `submits=[]`, `recovers=['task-chunk_01']`,
  `chunk_01.mp3` создан, run `completed`, попытка `completed`/`cost=0.3` —
  **второго POST нет**.

**Часть 2 — stand-alone Groq `timings` raw-after-POST → локальный replay.**

- Failpoint: fake `transcribe_timing_audio` отдал тело в `on_raw_response(...)`
  (raw сохранён до parse) и затем бросил `requests.Timeout`.
- Наблюдение: `timings --timing-provider groq-whisper --json` → exit `40`;
  run `running`, attempt `raw_saved`, число POST = 1.
- `history resume <uuid>` → exit `0`, `status=completed`, `complete=true`;
  число POST осталось **1** (replay из сохранённого тела, без второго запроса).

**Часть 3 — локальные ASR/`verify` метаданные.**

- `transcribe --provider qwen-local --json` → exit `0`,
  `history.saved=true`; `verify-tts --expected-text ... --json` → exit `0`,
  `passed=true`, `history.saved=true`, `human_listening_required=true`.
- Итоговые операции истории: `tts/completed`, `timings/completed`,
  `asr/completed`, `verify/completed`.

Эти результаты — про **семантику записи и восстановления**, а не про качество,
слышимость или реальный биллинг: fake-провайдеры и синтетический звук не являются
доказательством воспроизведения. Детерминированный эквивалент каждой части лежит
в `tests/test_history_native_generation.py` (`...resume_known_id...`),
`tests/test_paid_transcription_timing.py` (`...raw_saved_after_artifact_failure...`)
и `tests/test_history_native_quality.py`/`tests/test_asr_history.py`.

## Не доказано и ограничения

- **Live/provider/local-model/listening NOT_RUN**: ни одного настоящего
  provider/ASR-запроса, биллингового сравнения, прослушивания или загрузки весов
  не выполнялось. S01 остаётся source-only/`BLOCKED_PROVIDER_CONTRACT`.
- **Installed-model NOT_RUN**: реальные audio.cpp/Qwen/OmniVoice/faster-whisper
  запуски не выполнялись; проверялась только запись/восстановление через швы.
- Cloud ASR (`polza` Qwen ASR) — S07, не S05: сейчас не реализован, ничего не
  сохраняет.
- Точный upstream-контракт `openrouter-whisper` и облачного `xai-stt` на живом
  провайдере не подтверждался.
- Codebase/Serena/ast-grep в этой сессии недоступны; приёмка маршрутности
  опирается на исходники и офлайн-тесты.
- `ACCEPT_OFFLINE` не отменяет отсутствие live/provider/local-model/listening
  evidence, а Sol6 BLOCK относится к версии до исправления output-root P1;
  post-fix повторный независимый review **NOT_RUN**. Предыдущий разрешённый push
  довёл `origin/main` до `c0c56b6`; дальнейший push/tag/release/publication —
  **NOT_RUN** и требуют отдельного разрешения.
