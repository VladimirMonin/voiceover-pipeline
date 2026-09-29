# S03 — Выделение исполнения CLI в сервисы

| Поле | Факт |
|---|---|
| Дата / ветка | 2026-09-29 / `main` |
| Принятый кодовый HEAD | `f373d546d5708bc6854bb5d0afb28404632597b1` — `fix: match dialogue cast voice during paid raw resume` |
| Статус | **ACCEPT_OFFLINE** по S03 основного [плана](../plans/2026-09-29-voiceover-pipeline-development-plan.md); реальные provider-контракты, локальные модели и прослушивание — **NOT_RUN** |
| Область | Только локальные commits; без push, tag, release, платных/live запросов приложения, загрузки моделей и чтения секретов |

## Кодовые SHA и изменённые файлы

Ниже все 19 локальных кодовых commits S03 после `5e3f7079572726c1de362149f9ceed69f43ed059` (инструкционный commit не входит в S03). В таблице `C` — `src/voiceover_pipeline/cli.py`, `S/` — `src/voiceover_pipeline/services/`, `K/` — `src/voiceover_pipeline/commands/`, `T/` — `tests/`. Имена в каждой строке относятся именно к указанному commit; полный диапазон меняет **23 файла** в `src/voiceover_pipeline/` и `tests/`.

| Commit | Граница | Изменённые файлы |
|---|---|---|
| `dac01261aa3ebca1a7633a015216e653799447b7` | Подготовка split | `C`, `K/__init__.py`, `K/split.py`, `T/test_cli_json_contract.py` |
| `7c5d8cfca6ad0cf2ff71168cfb9cab9856afac79` | ASR request/validation | `C`, `S/__init__.py`, `S/transcription.py`, `T/test_asr_cli.py` |
| `2c80f7e4458863f3d667b6ad82e58d51d16a7e46` | `PreparedRun`/`PreparedPart`, TTS seam | `C`, `S/prepare.py`, `S/synthesis.py`, `T/test_generation_stability.py` |
| `a95e2b524a04dd3b532106e49b3ae413e4017b25` | Проекции прямой точной цены | `C`, `S/costs.py`, `T/test_cli_json_contract.py`, `T/test_pricing.py` |
| `0b32bde363477571c143fd784cd59ded32e88fac` | Read-only eligibility recovery | `C`, `S/recovery.py`, `T/test_cli_json_contract.py`, `T/test_generation_stability.py` |
| `396eda1a785e12ca7b0a62c4c72a3ad1386a8c4e` | Cost-history enrichment | `C`, `S/cost_enrichment.py`, `T/test_pricing.py` |
| `aaa7c26933550555de06faf41a8484b590fe1e5e` | Script fragments и OmniVoice preparation | `C`, `S/prepare.py`, `T/test_prepare_service.py` |
| `c83d685135779e60785ace5a749175f02aedb3e7` | Paid receipts и Media helpers | `C`, `S/execution.py`, `T/test_execution_service.py` |
| `527dfa7e0e83878e73d0e563256ebfc05c8c7592` | Единый paid part lifecycle loop | `C`, `S/execution.py`, `S/recovery.py`, `S/synthesis.py`, `T/test_execution_service.py`, `T/test_generation_stability.py` |
| `0029fdb9e98bcc10c9fc0c83156aafad50c0fb5f` | ASR backend invocation | `C`, `S/costs.py`, `S/transcription.py`, `T/test_asr_cli.py` |
| `247c1e74eaaa218797cfb7f9aae4c0d96251d535` | Pre-loop state/resume setup | `C`, `S/execution.py`, `T/test_execution_service.py` |
| `6ddfe76cdc7e40f9edf5bf0e616aec210fc0bd06` | ASR timing и dialogue-quality gate | `C`, `S/transcription.py`, `T/test_asr_timing_bridge.py`, `T/test_gemini_dialogue_e2e.py` |
| `b2a57907078b0d98c23d7942c556f7fe2c87f184` | Decimal run total и trusted state merge | `C`, `S/cost_enrichment.py`, `T/test_pricing.py` |
| `da90ee7f583f5014fa4a561c81c414303830aafa` | Post-loop finalization/artifacts | `C`, `S/finalization.py`, `T/test_finalization_service.py` |
| `8ab846c0e714756155deb2d80119060e2ec69444` | Запрет повторного OpenRouter submit после `TypeError` | `S/synthesis.py`, `T/test_generation_stability.py` |
| `12b4be60442c8bc9ee306f51df67320f9e366c52` | Generic ASR timing execution | `C`, `S/transcription.py`, `T/test_asr_timing_bridge.py` |
| `c947e0d18be3138cd17e97d3cdb6f3ea625ab78a` | Provider factory и dialogue cast providers | `C`, `S/provider_factory.py`, `T/test_provider_factory_service.py` |
| `391bcf1c3380074fd8d9d67593fdaffe8cd097a7` | Generation identity/style и pricing route | `C`, `S/prepare.py`, `S/cost_enrichment.py`, `T/test_prepare_service.py`, `T/test_pricing.py` |
| `f373d546d5708bc6854bb5d0afb28404632597b1` | Cast voice раннего paid raw resume | `C`, `T/test_gemini_dialogue_e2e.py` |

Этот отчёт и изменение индекса — отдельный последующий **документационный** commit, не часть принятого кодового HEAD.

## Доказанные офлайн-контракты

- Parser, старые CLI import-пути и совместимые monkeypatch-точки сохранены тонкими wrappers и call-time hooks. Аргументы, машинный JSON/exit и порядок важных событий покрыты CLI/JSON/regression tests; `argparse` не заменялся. `cli._generate_step` делегирует подготовку state/resume, **один** упорядоченный цикл `PreparedPart` и финализацию сервисам, затем оставляет вывод JSON/human. Formatter/manifest builders не делают HTTP; provider lookup остаётся отдельной границей.
- TTS provider выбирается до одного вызова части; OpenRouter legacy-сигнатура определяется **до** обращения к провайдеру. `TypeError` после вызова не вызывает запасной повторный submit. Локальные optional runtime импортируются лишь на выбранном маршруте; существующие локальные провайдеры не становятся обязательными для всех CLI-команд.
- Ранее принятый [S02 paid-safety контракт](2026-09-29-s02-paid-safety.md) сохранён: marker до paid POST, неопределённый outcome блокирует повтор, принятые raw bytes/receipt до FFmpeg, целый raw восстанавливается локально; известный Polza Media ID допускает только GET recovery той же identity. Новая регрессия полного CLI пути доказывает, что dialogue без `--voice` сравнивает валидированный первый cast voice с state, восстанавливает свой raw **без synthesis, Media GET или POST**, но чужая сохранённая voice блокируется до key/provider/pricing. `--overwrite` при marker по-прежнему закрыт.
- ASR request, provider invocation, generic timing bridge и dialogue-quality ASR выполняются через `services.transcription`; quality gate предшествует concat. После цикла `services.finalization` присоединяет позднюю наблюдённую стоимость, пишет артефакты и completed state **до** `run_complete` log/JSON. Канонический total складывает exact-строки через `Decimal`, наблюдённый ноль не превращается в неизвестную цену; state chunk сверяется по **id и number**. Поздний mismatch не записывает state на диск, хотя ранее совпавшие элементы могли уже измениться **в памяти** до исключения.

## Проверки на принятом кодовом HEAD

- Регрессии были воспроизведены без реальных запросов: OpenRouter post-call `TypeError` до исправления давал **2** вызова, после — **1**; ранний dialogue raw-resume до исправления давал exit **30** для своего `Kore` и не блокировал чужой `Puck` до key, после исправления свой raw восстанавливается без provider вызова, чужой получает `PAID_SUBMIT_UNCONFIRMED` до key/provider/pricing. Часть новых сервисных тестов также краснела на отсутствующих функциях. Имитацию отсутствующего готового provider-factory модуля не считаем настоящим red-before-implementation.
- Родитель: `uv run --offline --frozen pytest -q tests/test_gemini_dialogue_e2e.py tests/test_generation_stability.py tests/test_execution_service.py tests/test_cli_json_contract.py` — **173 passed**; `uv run --offline --frozen pytest -q -p no:cacheprovider` — **1052 passed, 2 skipped**.
- Родитель: `uv run --offline --frozen ruff check src tests` — PASS; `ruff format --check src tests` — **117 files**; `mypy --no-incremental` — PASS, **68 source files**; `git diff --check` — PASS. Перед scoped кодовым commit индекс был пуст; staged diff SHA-256 для последнего P1-fix: `85aec3413555851029f9b096a6fd01ccb515dc856bbe37e675b926cb0c2e56f2`. После commit tracked tree/index чисты, только пользовательские `?? .pi/` и `?? .serena/` не включены.
- Независимые Sol6 reviews по мере переноса фиксировали **PARTIAL/BLOCK** до закрытия реальных границ и двух P1. Финальный fresh read-only recheck `9b6c77a7-ef68-4870-9d76-c13616a2ebfd` на родительски подтверждённых байтах `f373d54…`: **ACCEPT_OFFLINE, no issues found**. Reviewer сам не мог запускать тесты/Git/Codebase/Serena/AST и не подтвердил commit-range/HEAD независимо; это родительская проверка. Долговечное правило не менялось, обновление `instructions/` не требуется.

## Не доказано и известные границы

- **Live/provider/local-model/listening NOT_RUN**: моки и source recheck не доказывают ответы и billing реальных Polza/OpenRouter API, фактический MIME/голоса, звук, качество прослушивания, работу/доступность моделей или ускорение. Полная S01 live/listening-приёмка остаётся `BLOCKED_PROVIDER_CONTRACT` до отдельного разрешения и бюджета владельца; её source-only [отчёт](2026-09-29-s01-source-only.md) не является live PASS.
- Сеть и диск не атомарны вместе; сбой между оплаченным ответом и raw receipt может оставить неразрешённый исход без восстанавливаемых байтов. Сохранённые bytes и guarded resume доказаны только для покрытых офлайн-сценариев. Отсутствующие плановые имена S02 log-событий `retry_scope` и `submit_outcome_unknown` остаются известным [пробелом S02](2026-09-29-s02-paid-safety.md), а не новым S03-доказательством.
- Существующий `--dry-run-cost --json` для dialogue без явного `--voice` выходит **до** cast binding и может показать provider default вместо первого cast voice. S03 сохранил прежний JSON-контракт, но не утверждает, что этот отдельный projection исправлен. Новые speech-parts/voice/vibe и их live/listening приёмка относятся к S06; S04–S12 и финальная S11-матрица не объявляются выполненными данным этапом.
