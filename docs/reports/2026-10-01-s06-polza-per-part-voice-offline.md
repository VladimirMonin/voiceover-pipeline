# S06 — подготовка привязки голоса Polza к каждой части (офлайн)

Дата: 2026-10-01. База `main` `4c5c753698ab81c51035a48a29e884c30f7c733c`. Статус: **ACCEPT_OFFLINE_PREREQUISITE**, но **Gemini 3.8 всё ещё `BLOCKED_PROVIDER_CONTRACT`**; live/стоимость/прослушивание `NOT_RUN`. Это не принятие полного S06 и не разрешение на POST.

## Что изменилось

| Файл | Проверенный Git blob |
|---|---|
| [`providers/polza_tts.py`](../../src/voiceover_pipeline/providers/polza_tts.py) | `b5472d1116791e0f95848d35030d717477252e0b` |
| [`services/native_generation.py`](../../src/voiceover_pipeline/services/native_generation.py) | `0dec1e0227d21202e1f9165b754548efe1d28626` |
| [`test_history_native_sync_response.py`](../../tests/test_history_native_sync_response.py) | `7cb97d04c58aa6ef5f804c4245e9f93c6233c78c` |

- Синхронный адаптер `polza-tts` принимает опциональный `voice` для одной части, передаёт выбранный голос в JSON `/audio/speech` и сохраняет его в результате. Без override используется прежний голос запуска. ElevenLabs `/media` при попытке передать **иной** голос отказывает до POST, а не молча игнорирует его.
- Native private pre-parse HTTP body/receipt теперь связывает запрос с `part.effective_voice`; script-free локальное восстановление сверяет и парсит ответ с **тем же** голосом. Для прежних обычных прогонов эффективный голос равен голосу запуска, поэтому старые receipt остаются совместимыми. Несовпавший receipt блокирует восстановление, не создавая новый POST.
- `speech_parts_route.py`, регистрация моделей, CLI admission и перенос `vibe` **не менялись**. Пример `Kore`/`Puck` в тестах — синтетические метки при test-only monkeypatch допуска маршрута; они **не** подтверждают голоса Polza для GPT-4o или Gemini 3.8.

## Доказательства и границы

Рабочий `worker` `1612c674-8243-4b8b-bb46-6672ad37480d` добавил четыре fail-first теста: до правки **4 failed**, после **4 passed** (детали в `s06-polza-voice-offline-worker.md`). Среди них два fake HTTP POST с разными голосами/приватными receipts и fault injection после второго сохранённого ответа с восстановлением из SQLite **без третьего POST и без исходного YAML**. Никаких реальных POST или чтения ключа не было.

Родитель на окончательных байтах запускал тесты из временного CWD вне checkout с синтетическими процессными ключами и `uv --offline --frozen`:

- Фокусно три файла тестов: **143 passed**.
- `pytest -q -rs -p no:cacheprovider` весь `tests/`: **2378 passed, 2 skipped** (native Windows lock на macOS, WVM reference без `WVM_ROOT`).
- `ruff check src tests`: PASS; `ruff format --check src tests`: PASS (**193 files**); `mypy --no-incremental`: PASS (**92 source files**); `git diff --check`: PASS.

Независимый fresh-context Sol6 reviewer `7c9df7e8-b0d5-4e86-a824-bf4b8c22817e` на этих исходниках: **OK with notes**, конкретных дефектов не нашёл. Его тесты/Ruff/mypy **NOT_RUN** из-за ограниченного набора инструментов, внешние Polza-документы не проверялись; родительские гейты выше выполнены отдельно. Исследование: Codebase index `Users-v-Documents-py-voiceover-pipeline` → Serena `_bind_sync_response_sink`, `_replay_sync_response`, `_synthesize_audio_speech` и ссылки на `verify_sync_response_receipt` → worker ast-grep по `part.effective_voice`/прежнему `prepared.voice`. Worker не имел Codebase/Serena в своём allowlist и честно сообщил этот gap; доступной Kanban-доски нет.

## Следующий предел

[Уточнение тарифа и внешнего контракта](2026-10-01-s06-gemini-contract-preflight.md) не нашло подтверждённых Gemini 3.8 голосов, поля инструкции и модельно-специфичного ответа. Кандидат остаётся вне `list` и отказывает **до ключа/POST**. Для paid probe требуется авторитетный модельный контракт либо явное отдельное решение владельца разрешить исследовательскую пробу с неподтверждённым голосом/форматом в прежнем лимите; до этого никаких новых сетевых запросов. `--vibe`, два реальных голоса, контейнер и звучание не объявляются принятыми. Изменение инфраструктуры голосов не заменяет live-приёмку или S12.
