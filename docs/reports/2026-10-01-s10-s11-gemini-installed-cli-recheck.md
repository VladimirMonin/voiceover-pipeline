# Установленный CLI на пяти сохранённых Gemini-пробах — offline recheck

Дата: 2026-10-01. Статус: пакет/ключе-независимое чтение SQLite доказано офлайн; live/paid, локальное восстановление и релиз **не выполнялись**. Продолжение проверки [S08 saved search](2026-10-01-s08-gemini-saved-search-recheck.md); пять прогонов остаются в пяти отдельных приватных `VOICEOVER_HOME`, не в одном общем хранилище.

## Пакет

На текущих байтах после исправления read-only history повторно собран `uv build --wheel --offline`: `voiceover_pipeline-0.6.1-py3-none-any.whl`, SHA-256 **`fd67b03eb9406182d6a66ee878389f206720f63e3876b480a05b9e1dfdcf37ac`**. Установлен в отдельное `/private/tmp/voiceover-packaged-cli.Zv19Qf/venv` (`uv pip install --offline --no-deps`), импорт `voiceover_pipeline` подтверждён из этого venv, не из checkout. Недостающие для полностью самостоятельного clean-install зависимости получены **только** через добавленный после installed site-packages путь уже имеющегося проектного `.venv` без обработки editable `.pth`: это не доказательство `uv pip install --offline` со всеми зависимостями. Версия по-прежнему **0.6.1**; S12 и публикация не разрешены.

## Обнаруженный и исправленный отказ

Первый установленный wheel после S08 `index build` возвращал `history list --json` exit **30** / `HISTORY_DATABASE_UNREADABLE`: в пяти private DB после записи индекса оставалась легитимная нулевой длины пара `history.sqlite3-wal`/пригодный `history.sqlite3-shm`. Старый `list/show` всегда использовал immutable reader и отказывался от любого sidecar, хотя `history costs` и `index status/search` уже читали WAL-консистентно. Исправление переключило `list/show` и native read-only preflight на существующий проверенный `connect_readonly_consistent`: без WAL — immutable без новых sidecar, с WAL и пригодным уже существующим SHM — чтение закоммиченных кадров через `mode=ro`; отсутствующий/непригодный SHM, journal, symlink, чужой ledger остаются fail-closed. Отдельные fail-first тесты проверили live WAL и пустую WAL-пару с `history list/show` плюс прежний отказ при WAL без SHM. Основной SQLite-файл и WAL не меняются; SQLite **может обновить байты существующего SHM** при блокировке чтения, поэтому обещание побайтовой неизменности SHM снято из [машинного контракта](../agent-cli-contract.md). Независимый Sol6 reviewer дал `ACCEPT with note`: `history show` мог смешать старый статус run с новыми attempts при конкурентной WAL-записи. Дополнительно введена единая deferred read-транзакция lookup+detail с fail-first тестом на commit другого writer между запросами. Никаких sidecar вручную не удаляли.

## Повторная проверка установленного пакета

В изолированном процессе с `socket.connect`/`socket.create_connection` fail-closed, из CWD вне checkout, без чтения `.env` агентом и без запросов провайдера:

- `voiceover help speech.parts|search.lexical|runs.resume|providers.polza --json`: 4/4 темы доступны из установленного wheel; `list providers --json` — success, регистрация ≠ live-доступность.
- Для каждого из **пяти** приватных `VOICEOVER_HOME`: `history list` возвращает ровно один `completed` run; `history show UUID` — ровно соответствующие части/попытки, не выводит произнесённый текст; `history costs` — точное сохранённое `0 RUB`, `known_attempts` равно числу записанных attempts и `unknown_attempts=0`. Это ноль из API-ответов, **не** проверенная выписка Polza.
- `index status.complete=true`, ложный старый style-чанк отсутствует, слова речи находятся с `--scope speech`, реальные сохранённые per-part настроения рассказа/диалога/спектакля — с `--scope directions`, не в `speech`. Для Kore/Puck настроение не задавалось. `validate --format speech-parts --provider polza-tts --model google/gemini-3.8-flash-tts` вернул `route.admitted=false`; exit 0 у проверки синтаксиса **не** открывает генерацию.

| Изолированный прогон | `runs` | `parts` | `attempts` | `text_sources` | Направление найдено |
|---|---:|---:|---:|---:|---|
| Kore | 1 | 1 | 1 | 3 | не задавалось |
| Puck | 1 | 1 | 1 | 3 | не задавалось |
| Рассказ | 1 | 2 | 2 | 4 | да |
| Диалог | 1 | 2 | 2 | 4 | да |
| Мини-спектакль | 1 | 3 | 3 | 5 | да |

До/после всех read-only CLI вызовов SHA основного DB, упорядоченные статусы/стоимости attempts, канонические количества и наличие WAL/SHM остались прежними. Контрольный процесс не предпринял сетевых вызовов (`0`), новых POST (`0`). `history export` как отдельной leaf-команды нет: JSON-манифесты — совместимые локальные экспорты генерации/`sync`, а `sync` может выполнить GET по известному Media ID; **ни `history sync`, ни потенциально оплачиваемый `history resume` в этом recheck не запускались**. Проверка внешнего тарифа и второго голоса **в одном POST** остаётся `NOT_VERIFIED/NOT_RUN`.

Финальные автоматические гейты на изменённом Python-срезе: `uv run --offline --frozen pytest -q -o addopts=''` — **2390 passed, 2 skipped**, Ruff lint/format PASS (**193 files**), mypy `--no-incremental` PASS (**92 source files**), `git diff --check` PASS. Первый независимый review WAL-read — `ACCEPT with note`, после fail-first исправления P2 повторный независимый Sol6 recheck — **ACCEPT / no issues** (reviewer не запускал тесты, не проверял приватные DB и wheel SHA). Окончательный scoped commit выполняется отдельно; release/публикация не разрешены. Сборка и smoke offline, версия/release не обновлялись.
