# S06 addendum — private Polza response before parse, local replay

| Поле | Факт |
|---|---|
| База | `main` на `69ba07c499d9f81f7fb4f2d051c889daf41c65b3`; исторический [S06 offline-core отчёт](2026-09-30-s06-speech-parts.md) остаётся источником по speech-parts. |
| Статус | **ACCEPT_OFFLINE_SAFETY / BLOCKED_PROVIDER_CONTRACT**. Native `polza-tts` `/audio/speech` не теряет полученное тело до parse и восстанавливает его без второго POST. Это **не** live-приёмка Gemini 3.8 и не подтверждение цены/голосов/instruction. |
| Git | Локальный scoped commit после финальных gates; неизменяемый SHA сообщается отдельно. Push/tag/release **не выполнялись**. |
| Live | В этом срезе: GET **0**, POST **0**, listening **NOT_RUN**; новый Gemini-маршрут не регистрировался как stable. В рамках ранее разрешённого эксперимента суммарно был **один** GET `/models`, **ноль** POST. |

## Изменённые файлы и контрольные Git blob SHA

| Файл | Blob после исправления Sol6 P1 |
|---|---|
| `src/voiceover_pipeline/providers/polza_tts.py` | `fef58eaed0c08e99c74870f7ffa2ecf74cd168a6` |
| `src/voiceover_pipeline/history/raw_receipt.py` | `75eca1ceabfb1bf8c932243fc58f3e8c06b335c9` |
| `src/voiceover_pipeline/services/native_generation.py` | `34ebb9a9757ddb98b19bb9bc3130ac423c35ef88` |
| `tests/test_history_native_sync_response.py` (new) | `997126fba7142e60227900baf4a557a2e7d0e751` |
| `instructions/provider-cli.instructions.md` | `607dd6f71097364ec5e53fce8f858a2ec92be4a5` |
| `docs/agent-cli-contract.md` | `55a4061a8b25e0b11959d207cad67057e6e415f8` |
| `docs/README.md` | Индекс этого отчёта, blob после записи. |

Применены [AGENTS.md](../../AGENTS.md) и маршруты `core`, `code-intelligence`, `agent-kanban`, `git-release-safety`, `provider-cli`, `test-quality`, `docs-governance`, `instruction-authoring`. Родитель искал seam через Codebase (индекс `Users-v-Documents-py-voiceover-pipeline`), затем Serena/адресные symbol reads; Codebase-инструмент дал `_link_sync_raw`, `_submit_fresh_part`, `_reconcile_sync_raw_receipt` и `PolzaTTSProvider._synthesize_audio_speech`. Worker в своём tool allowlist не имел Codebase/Serena/ast-grep и сообщил это как ограничение. Родитель дополнительно проверил AST-паттерны `self.on_raw_response($$$ARGS)` в `providers/polza_tts.py:212–216` и `verify_sync_response_receipt($$$ARGS, response_format=$FORMAT)` в `services/native_generation.py:3006–3017`; порядок исполнения и идентичность доказаны адресным чтением, тестами и review, а не одним AST-совпадением. Kanban board tool недоступен, `todo`/goal tracking — актуальная отметка.

## Что доказано офлайн

1. **Pre-parse paid boundary.** После durable native SQLite-маркера `submitting` и единственного синтетического HTTP POST callback получает тело до HTTP status check, JSON parse, base64 и определения контейнера. Приватный `raw/<attempt_uuid>.response` (≤16 MiB, файл 0600, каталог 0700) и атомарный bounded receipt связывают UUID попытки/части, chunk id/number, fingerprint, provider/model/voice/**request response format**, код HTTP, digest/size и только очищенный opaque header ID. Ошибка записи/пустое или превышающее лимит тело не включает новый paid retry; HTTP-error body остаётся приватным и не попадает в публичную диагностику.
2. **Только локальное восстановление.** На resume `submitting` сначала ищется любая половина response-evidence. Полный совпавший 2xx body/receipt разбирается **без provider, ключа, GET/POST**, до decoded-raw-only fallback. Если decoded raw и его receipt уже существуют, они сверяются с тем же ответом, сохраняется наблюдённая Decimal cost, затем конвертация; crash после raw receipt до DB-link теперь не превращает известную стоимость в `NULL`. Изменённый format/model/identity, чужой raw, частичное или повреждённое evidence, HTTP error и malformed body → `PAID_SUBMIT_UNCONFIRMED`, без автоматического повторного запроса. Старые прогоны **без** response-evidence могут по-прежнему восстановиться из decoded raw с неизвестной стоимостью, не придумывая её.
3. **Декодер без отражения данных.** Общий parser для live и replay отвергает нестроковый/пустой/невалидный base64 `audio`, не декодирует WAV без RIFF, сообщает фиксированные ошибки вместо echo response keys, `contentType`, текста/URL или Authorization. Небезопасный `X-Generation-Id` отбрасывается; допустимый ID из body приводит live/replay к одному bounded результату. Unquoted usage разбирается как `Decimal`.

## Тесты и независимый review

- Worker `eb30bdc7-76a8-4880-9b32-9f6459abdcf3` подготовил 13 focused тестов; из временного CWD вне репозитория с четырьмя синтетическими key env-переменными его подмножества дали **483** и **239** passed. Никакого обращения к пользовательскому `.env`, провайдеру или модели.
- Первый Sol6 read-only review `2354841c-671a-4874-a5c6-37ab0275056d` — **BLOCK**: P1 format identity не проверялась, P1 стоимость терялась в later decoded-raw/DB crash window, P2 truthy object `audio` давал generic error. Три новых теста сначала дали **3 FAILED** именно по этим наблюдаемым дефектам; затем исправлены. Добавлены два регрессионных теста на запрет echo приватных JSON keys/contentType и на одинаковый generation ID в live/replay. Focused после fixes: **234 passed**.
- На исправленном коде полный offline/frozen pytest с временным CWD, синтетическими ключами и без `.env`: **2226 passed, 2 skipped** (native Windows file locking на macOS и WVM reference без `WVM_ROOT`). `ruff check src tests` PASS, `ruff format --check src tests` PASS (173 файла), `mypy --no-incremental` PASS (91 source files), `git diff --check` и отдельная whitespace-проверка нового теста PASS.
- Fresh-context Sol6 recheck `284c436a-2a4d-4d5d-a091-89f0f9a775d7` подтвердил закрытие всех трёх замечаний, новых дефектов не нашёл: **ACCEPT_OFFLINE**. Reviewer **не запускал** pytest/линтеры и **не проверял** live-provider.
- После записи отчёта и ссылки из docs index полный offline/frozen pytest повторён: **2226 passed, 2 skipped**; Ruff lint/format (173 файла), mypy (91 source files), tracked/new-file whitespace — PASS. Проверено **48** относительных ссылок трёх docs, instruction frontmatter/router и все шесть указанных Git blob SHA — PASS. Последний повтор после этой записи идёт до commit; `logged_events_checked`: **NOT_RUN** (события replay/parse-error не проверены как отдельный контракт).

## Внешний эксперимент остаётся заблокированным

Единственный ранее разрешённый приложению GET `/models`: HTTP 200, SHA-256 каталога моделей `131989c9fff80c4444b552efaea79e28965e7e6bad511c6d9e69128330710b95`. ID `google/gemini-3.8-flash-tts` и `google/gemini-3.8-flash-lite-tts` подтверждены. Для Flash наблюдались RUB prompt `58.16300000`/1M и completion `1046.93400000`/1M tokens; для Lite — completion `697.95600000`/1M. Поле `tts_per_million_characters` существует, но его **значение не вошло** в фильтр вывода, а raw GET-body не был сохранён. Применимый тариф/верхний предел ≤200 ₽, endpoint, instruction field, поддерживаемые голоса, контейнер и формат ответа пока **не доказаны**. Дополнительный GET запрошен у владельца, но **не разрешён**; quota одного GET исчерпана. Не делать POST до доказуемой cumulative reservation ≤200 ₽/≤5 POST и безопасного контракта; не объявлять fake-тест подтверждением звучания. Приложение может читать пользовательский ключ только для отдельно согласованных запросов, агент содержимое `.env` не читает и не раскрывает. Polza Qwen ASR, downloads и push также вне разрешения.

Остаточные риски: `requests` буферизует тело до проверки storage cap 16 MiB (memory-bound streaming ещё не доказан); старый legacy TTS executor и `openrouter-tts` не имеют этого pre-parse response sink; GET shape, paid billing и listening остаются `NOT_RUN`. Поэтому полный S06 **не принят**, даже если этот offline safety-срез принят.
