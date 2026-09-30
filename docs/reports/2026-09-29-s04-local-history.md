# S04 — Локальная SQLite-история и безопасный offline-import

| Поле | Факт |
|---|---|
| Дата / ветка | 2026-09-29 / `main` |
| Кодовый HEAD | `05f43e8774dd6c2adf4eee09e6157764d1e169a4` — история, importer и CLI |
| Проверенный HEAD с документацией | `025ff62bce73b6747bfd76ec644dab4a0038885d` |
| Статус | **ACCEPT_OFFLINE** для S04 основного [плана](../plans/2026-09-29-voiceover-pipeline-development-plan.md); provider/local-model/listening **NOT_RUN** |
| Граница Git | Только локальные scoped commits; **push, tag, release, публикации не выполнялись** |

## SHA и область изменений

База — принятый отчёт S03 `db2da8e9ecaf175784ca4b68888d262d2feaae83`. Ниже перечислены **все 14 изменённых файлов** четырёх локальных commits S04; `H/` означает `src/voiceover_pipeline/history/`, `T/` — `tests/`.

| Commit | Срез | Изменённые файлы |
|---|---|---|
| `d3fd4f7fcc1fec5b4dca0c9dfaf90aa0a90c2751` | Schema, repository, migration | `H/__init__.py`, `H/database.py`, `H/repository.py`, `T/test_history_database.py`, `T/test_history_repository.py` |
| `7ed92da370df2e017f08ed3c1b940d59efb3fec1` | Private home, offline importer | `H/__init__.py`, `H/legacy_import.py`, `H/paths.py`, `T/test_history_legacy_import.py`, `T/test_history_paths.py` |
| `05f43e8774dd6c2adf4eee09e6157764d1e169a4` | `history list/show/import` | `src/voiceover_pipeline/cli.py`, `src/voiceover_pipeline/commands/history.py`, `H/database.py`, `H/legacy_import.py`, `H/repository.py`, `T/test_history_cli.py`, `T/test_history_database.py`, `T/test_history_legacy_import.py`, `T/test_history_repository.py` |
| `025ff62bce73b6747bfd76ec644dab4a0038885d` | Машинный и skill-контракты CLI | `docs/agent-cli-contract.md`, `docs/skills/voiceover-pipeline/docs/06-commands-and-flags.md` |

Отчёт и ссылка в `docs/README.md` — отдельный последующий docs-only commit. `?? .pi/` и `?? .serena/` — пользовательские untracked каталоги, в scoped commits не включались.

## Доказанные офлайн-контракты

- **Schema/migrations.** Version/checksum/future-schema guard проверяется до WAL/DDL; ledger и миграции транзакционны, несовместимое обновление требует backup без перезаписи существующего backup. FK связывает части/попытки/артефакты внутри одного run. Repository round-trip сохраняет snapshots с редактированием секрет-подобных значений, текстовые источники и `Decimal`/raw денежный provenance. `NULL` неизвестной стоимости отличен от наблюдённого нуля; legacy JSON numeric lexeme сохраняется без ложного `exact_available: true`.
- **Пути и приватность.** История живёт в системном data directory независимо от CWD либо в абсолютном `VOICEOVER_HOME`; относительные/недопустимые/symlink paths отвергаются. Управляемый POSIX-дом создаётся `0700`, существующий открытый группе/миру дом не используется для plaintext-БД. База **не зашифрована** и намеренно хранит доступные текстовые источники; публичные CLI-проекции их не выводят.
- **Legacy import.** `history import DIR --dry-run --json` не создаёт БД, каталогов или sidecar и не меняет оригиналы. Нечитаемая, foreign/future либо WAL/journal БД даёт статус **unknown**, а не выдуманное отсутствие конфликта. Symlink-ancestor, скрытые/опасные script/chunk/audio/manifest paths, неполные тексты и пропавшее аудио обрабатываются fail-closed либо явно отмечаются. Один импортируемый run со всеми parts, attempts, artifacts и text sources заполняется в одной транзакции; повтор того же `legacy_source_root` не дублирует расходы/источники. Одинаковые метки из разных roots дают разные внутренние UUID.
- **CLI metadata/exit.** `list`/`show` на отсутствующей БД не создают её, `show ID` ищет UUID и затем точную метку; неоднозначность сообщает bounded candidates, не выбирает произвольный run. `list` фильтрует и ограничивает страницу. `--json` даёт один объект в stdout; существующие numeric exits `0/2/30/50` и стабильный `details.error_code` сохраняются. Reader `mode=ro&immutable=1` не создаёт sidecar и отказывает при активном WAL/journal, когда снимок мог бы быть устаревшим. Вывод не раскрывает сценарий, prepared text, transcript, raw snapshot, signed URL, Authorization header, произвольный remote ID или сохранённые ошибки; synthetic Basic/Digest/Proxy-Authorization регрессии сначала воспроизводили утечку, затем стали зелёными. Публичный cost показывает exact amount/zero/null и provenance, не превращая неизвестное в ноль.
- **Граница исполнителя.** S04 не переключает текущие `generate`/ASR/timing/verify на SQLite: прежние JSON остаются единственным writer текущей генерации до S05. Миграция и импорт не вызывают TTS/ASR, модель, сеть или платный provider. Content-free события `migration_applied`, `history_imported`, `history_conflict` имеют регрессионные тесты.

## Фактические проверки

- На кодовом срезе родитель запускал fail-first регрессии Basic Authorization (сначала **2 fail**, затем четыре дополнительных Digest/Proxy варианта **4 fail / 2 pass** до обобщения фильтра), после исправления — **263 focused passed**, **1260 passed, 2 skipped** во всём `uv run --offline --frozen pytest -q -p no:cacheprovider`. На принятом HEAD с документацией повторён итоговый gate: `uv run --offline --frozen pytest -q -p no:cacheprovider tests/test_history_database.py tests/test_history_repository.py tests/test_history_paths.py tests/test_history_legacy_import.py tests/test_history_cli.py tests/test_cli_json_contract.py` — **284 passed**; весь offline suite с `-rs` — **1260 passed, 2 skipped**. Skips: native Windows file locking на macOS и WVM reference validation без настроенного `WVM_ROOT`.
- `uv run --offline --frozen ruff check src tests` — PASS; `ruff format --check src tests` — **128 файлов**; `mypy --no-incremental` — PASS, **74 source files**; `git diff --check` и `git diff --cached --check` — PASS на точных проверенных байтах. Документационные относительные ссылки разрешаются; четыре новых `json` примера парсятся; frontmatter в изменённых файлах отсутствует по их существующему стилю.
- Ручной **синтетический** console smoke через установленный локальный `voiceover` в `TemporaryDirectory`: `--dry-run` оставил home отсутствующим, import дал `imported_count=1`, повтор — `skipped_count=1`, list/show сохранили метаданные и lexeme `"0.1000"` без полного текста, SHA-256 всех оригинальных файлов до/после совпали, созданный POSIX home приватен. Тестовое аудио — фиктивные bytes, **не** доказательство воспроизведения/качества.
- Независимые read-only Sol6 reviews для schema `d4adfea0-f6f1-4bcc-9340-998b596c8b71`, importer `4e148156-cad5-4d09-a63f-2b47e6a17ba6` и CLI `23a0f1ed-0e6a-4c5b-8cd7-69f0931170f9` после P1 fixes дали `ACCEPT_CANDIDATE`. Финальный fresh Sol6 stage-review `205f159d-c5d2-4f2f-afe4-81900806e19d` — **ACCEPT_OFFLINE, no issues found**. Он не мог независимо проверить Git диапазон/HEAD; тесты, Git, Ruff/mypy, Codebase/Serena/AST — **NOT_RUN_BY_REVIEWER**, перечисленные результаты — проверки родителя.

## Не доказано, ограничения и следующие этапы

- **Live/provider/local-model/listening NOT_RUN**: никаких настоящих provider/ASR запросов, загрузок моделей, биллинговых сравнений, аудиопрослушивания или проверки контейнера/MIME не выполнялось. [S01](2026-09-29-s01-source-only.md) остаётся source-only / `BLOCKED_PROVIDER_CONTRACT` до отдельного разрешения, маршрута и бюджета владельца; моки S04 не заменяют live-приёмку.
- Активный WAL/journal блокирует read-only `list/show` (exit `30`); `--dry-run` сообщает `database_status_unknown`, но это **не** обещание, что фактический import можно сделать без устранения причины. Безопасный resume legacy каталога без подтверждённой identity этим этапом не доказан. Транзакционность одного run не равна атомарности сети и диска или всей коллекции импортируемых runs.
- Текущая генерация не использует canonical SQLite writer; crash/recovery/locking, безопасный повторный экспорт JSON, все новые попытки ASR/timing и identity — **S05**, не S04. Поиск/индексация, новые голоса и релизные гейты — последующие этапы. Унаследованный skill-справочник `docs/skills/voiceover-pipeline/docs/06-commands-and-flags.md` был длиннее собственного ориентира ≤300 строк ещё до S04; эта структурная задолженность не меняет проверенный CLI-контракт, но остаётся для docs hygiene.
- Build/install, tag, push, publication и S12 release — **NOT_RUN**; разрешение владельца на каждую такую операцию требуется отдельно. Статус S04 — **только офлайн-приёмка**.
