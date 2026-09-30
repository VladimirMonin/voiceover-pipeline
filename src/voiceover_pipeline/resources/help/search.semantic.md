---
topic: search.semantic
title: Semantic и hybrid поиск — DEFERRED
summary: Семантический и гибридный режимы в этой версии недоступны; честный отказ SEARCH_MODE_DEFERRED и план следующего релиза.
related: search.lexical, history.find
---

## Текущее поведение: честный отказ

```bash
voiceover search "текст" --mode semantic --json
```

```json
{"status": "error", "error": "Semantic and hybrid search are deferred to a later release; pass --mode lexical.",
 "code": 2, "details": {"error_code": "SEARCH_MODE_DEFERRED"}}
```

`--mode semantic` и `--mode hybrid` возвращают `SEARCH_MODE_DEFERRED` с exit `2`. Это не поломка и не «пока пустой результат»: режимы осознанно не входят в текущую версию. Тот же отказ приходит без явного `--mode`, когда `<CWD>/settings.toml` задаёт `[search] default_mode = "semantic"` или `"hybrid"`: режим читается из несекретных настроек и отклоняется до открытия базы и любых моделей. Невалидный `settings.toml` без явного `--mode` даёт вместо этого `SEARCH_SETTINGS_INVALID` (exit `2`), а явный `--mode lexical` побеждает и работает (см. `voiceover help search.lexical`).

## Чего в этой версии нет

- Нет embedding-адаптеров: ни локального `Qwen/Qwen3-Embedding-0.6B`, ни облачного `qwen/qwen3-embedding-8b` через Polza.
- Нет `sqlite-vec`, нет векторного индекса, профилей пространства и кеша эмбеддингов.
- Нет extras `search` и `search-local`, и установка базового пакета ничего не докачивает.
- Нет облачного индексирования: ключ Polza сам по себе никогда не разрешал и не разрешает отправку корпуса.
- Нет семантической выдачи и RRF/dedup: это не «незаметно выключено», а не реализовано.

Лексический офлайн-поиск при этом работает: `voiceover help search.lexical`.

## Где искать будущий дизайн

Проект следующего релиза (планировочная метка `v0.8.0`): `docs/plans/2026-10-01-semantic-search-next-release-plan.md` в репозитории. Там описаны оба backend, профили и приватность, явная индексация с бюджетом (`--allow-cloud` и `--max-rub`, `--max-requests`), `sqlite-vec` и гибридная выдача.

Этот план — намерение и контракт, а не работающая функция: пока он не реализован, ни один семантический запрос не выполняется и не оплачивается. Не считайте приведённые в плане команды доступными в этой версии.

## Что доказано офлайн

FTS5-индекс, роли текста, фильтры до `limit`, idempotent rebuild и отсутствие выдуманных таймкодов. Реальная семантическая релевантность, доступность моделей и облачные embeddings остаются `NOT_RUN` и требуют отдельного согласия владельца.
