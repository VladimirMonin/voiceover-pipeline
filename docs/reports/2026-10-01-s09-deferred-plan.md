# S09 — отдельный план следующего релиза (не v0.7.0)

| Поле | Факт |
|---|---|
| База | `f026dd25e16f08573f187b3d4306f0146ebf49e0` (`main`, после локального S07 commit) |
| Статус | **ACCEPT_PLAN_ONLY / DEFERRED / OUT_OF_SCOPE_FOR_V0.7.0**. S09 implementation `NOT_STARTED`; планируемое `v0.8.0` не является утверждённой версией или разрешением на релиз. |
| Основной артефакт | [Semantic search — next-release plan](../plans/2026-10-01-semantic-search-next-release-plan.md); Git blob `b7980f9dbe27316b15f8eeed678cfdb190db1bc3`. |
| Сопутствующие файлы | [Исходный план](../plans/2026-09-29-voiceover-pipeline-development-plan.md) (blob `501e1e954c4a95201822ab88833dc5c7e3dbf045`), [docs index](../README.md), [agent skill](../skills/voiceover-pipeline/SKILL.md) (blob `c4b26507db7c01fad989118b35eb4e6f2bd44075`). Код, тесты, lockfile и инструкции не менялись. |
| Git | Scoped локальный docs commit после финальных проверок; hash будет сообщён после создания. **Ни push, ни tag, ни release.** |

## Доказанный результат

S08 FTS5 остаётся обязательным для `v0.7.0`; `search --mode semantic|hybrid` по-прежнему даёт `SEARCH_MODE_DEFERRED`. Основной план теперь явно исключает S09 из S00–S08/S10/S11/S12-gate: S01 embedding-пробы, semantic self-check, S10 рабочую semantic-справку, embedding extras/конфиг/установку и проверку релевантности перенесли в следующий релиз. Тема `help search.semantic` в `v0.7.0` запланирована как **объяснение DEFERRED**, не работающий workflow.

Новый самостоятельный план задаёт два backend (локальный `Qwen/Qwen3-Embedding-0.6B`, Polza `qwen/qwen3-embedding-8b`), явную query/document подготовку, версионированные неизменяемые SQLite-профили/кеш, необязательный безопасный `sqlite-vec`, приватность без неявной загрузки корпуса, фильтры до top-k и hybrid RRF. Будущие **оба** облачных пути (build и query) требуют явного разрешения и `--max-rub`, build ещё `--max-requests`; durable marker и проверяемая reservation до POST, bounded private raw до parse, unknown без автоматического повтора, стоимость unknown = `NULL`. Это **контракты будущей реализации**, а не её доказательство.

## Проверки и review

- Docs-only проверка относительных ссылок пяти затронутых документов (включая этот отчёт) — **84 targets PASS**; YAML frontmatter навыка сохраняет `name` и `description`; три указанных blob SHA сверены с `git hash-object`; `git diff --check` и new-file whitespace — PASS после записи отчёта, до commit.
- Worker `556701ea-68f9-4ac3-aaba-0976524b38cd` сделал план и правки docs, без Git/network. Родитель устранил ограничения бюджета и release scope после независимых review.
- Sol6 `70ea4dbd-8b61-4fb9-b7e9-8d04b15f51c6`, затем `c795a1de-6301-4dc9-a5f8-8d3304ac3004`, затем `51db32c1-6689-4c01-8b9e-f66a4a9e5961` последовательно дали **BLOCK** на неполной бюджетной границе, неявной S09-зависимости S10/S01/матрицы и остатке embedding self-check. Все адресные замечания исправлены. Финальная fresh-context Sol6 проверка `20a01b4c-ee4f-40eb-aed0-b8c2289a46e4` дала **ACCEPT_PLAN_ONLY**, без новых замечаний; это **не** tech-review реализованного S09.
- Приложение не менялось, поэтому pytest/Ruff/mypy для этого docs-only commit отдельно **NOT_RUN**; предыдущие S07 gates не выдаются за проверку S09-кода. `logged_events_checked`: `NOT_APPLICABLE`.

## Границы

S09 планирует следующий релиз, но реализация, тесты semantic/hybrid и реальная локальная/облачная приёмка **NOT_RUN**. Независимый implementation plan-gate перед началом кода остаётся `OPEN` до отдельного решения владельца; review этого документа не разрешает скачивать веса, делать live/платные запросы или публиковать. `.env` не читался агентом, `.pi/` и `.serena/` пользовательские и исключены из commit. S12 v0.7.0 также не начинается без отдельного разрешения.
