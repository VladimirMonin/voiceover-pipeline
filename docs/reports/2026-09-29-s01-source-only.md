# S01 — Source-only отчёт по узким внешним контрактам

| Поле | Значение |
|---|---|
| Дата | 2026-09-29 |
| Ветка / HEAD | `main`, `1ba574341c1dc7f627ada124a26310229ccf1160` — `fix: resume accepted Polza Media tasks by ID` |
| Изменённые файлы | `docs/reports/2026-09-29-s01-source-only.md` (новый), `docs/README.md` (одна ссылка в таблице для разработчиков) |
| Характер документа | **SOURCE-ONLY**: без сети, live-вызовов, прослушивания, платных запросов, загрузки моделей и чтения секретов |
| Статус | Источник-только часть S01 — DONE. Полная S01 (live/listening) — **BLOCKED**, нужен явный approval владельца и бюджет на операцию |
| Реализация плана | Это не закрытие stage/release gate и не заявление о готовности релиза |

> Граница доказательств: ниже подтверждается только то, что статически видно в
> репозитории на `1ba5743…`. Ни один реальный контракт Polza/Google не доказан
> офлайн-кодом: реальные ID, endpoint, голоса, vibe, формат, цена и границы
> длины остаются `BLOCKED_PROVIDER_CONTRACT`. Live и listening — `NOT_RUN`
> (approval отсутствует). Ссылки плана `Wxx` — ранее просмотренные вторичные
> источники, **не** свежая проверка в этой сессии.

## Прочитанные источники истины

- `AGENTS.md` и routed-инструкции: `core`, `code-intelligence`, `agent-kanban`, `git-release-safety`, `provider-cli`, `test-quality`, `docs-governance`.
- `docs/plans/2026-09-29-voiceover-pipeline-development-plan.md`: §S01, таблица моделей (≈74–85), planned Gemini-интерфейс (≈229–302), Qwen (≈568–580), embeddings (≈633–650), source map W01–14 (≈1448–1493).
- Код: `src/voiceover_pipeline/providers/polza_tts.py`, `qwen_asr_local.py`, `asr_registry.py`; тест `tests/test_qwen_asr_provider.py`.
- `docs/reports/2026-08-21-gemini-dialogue-live-acceptance.md` — legacy OpenRouter **Gemini 3.1**, не доказательство для новой Polza Gemini 3.8.
- Ограничение инструментов: Codebase/Serena/ast-grep в этой worker-сессии не предоставлены; исследование ограничено явно названными файлами (fallback по условию задачи).

## Что подтверждено в коде (source-only)

- `PolzaTTSProvider` (`provider_id = "polza-tts"`) диспетчеризует по префиксу модели: `elevenlabs/*` → POST `{base}/media` (async, poll `GET /media/{id}`, download); **все остальные** → POST `{base}/audio/speech` (`model`, `input`, `voice`, `response_format`). `base_url = https://polza.ai/api/v1`, default `response_format = "mp3"`.
- В `PolzaTTSProvider` нет Gemini-3.8-специфичного mapping для общего/частного vibe, проверки незачитывания инструкции или предела длины. Пример `--vibe` в плане — **future syntax**, не действующий интерфейс Polza TTS.
- `src` не содержит `gemini-3.8` вообще; присутствует только legacy `google/gemini-3.1-flash-tts-preview` (config/openrouter/gemini_dialogue).
- Локальный `qwen-local` зарегистрирован с `QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"`; factory выбирает Python `qwen_asr` или настроенный audio.cpp. Python-адаптер берёт transcript из `response.text`, а word timestamps запрашивает с отдельным `Qwen/Qwen3-ForcedAligner-0.6B`; audio.cpp-адаптер берёт `local_response.transcript` из ответа runtime. Registry содержит только `QWEN_ASR_PROVIDER_SPEC` и `NEMOTRON_ASR_PROVIDER_SPEC`.
- `1.7B` ASR, Polza-cloud ASR-adapter и любой embeddings-adapter в `src` **отсутствуют**; `/embeddings` в `src` не встречается, `/audio/transcriptions` есть только у non-ASR-registry timing-провайдеров (`groq_whisper`, `openrouter_whisper`).

## TTS — облако Polza (Gemini 3.8)

| Модель | ID в плане | Реальный Polza ID/endpoint | Голоса | Общий+частный vibe | Инструкция не зачитывается | Формат | usage/cost | Граница длины | Source-only | Live | Listening |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash TTS | `google/gemini-3.8-flash-tts` (candidate) | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | PARTIAL: только generic dispatch `/audio/speech` | `NOT_RUN` | `NOT_RUN` |
| Gemini 3.8 Flash-Lite TTS | названа в плане/W11, **ID не задан** | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | `BLOCKED_PROVIDER_CONTRACT` | PARTIAL: тот же generic dispatch | `NOT_RUN` | `NOT_RUN` |

Примечания:
- Для обеих моделей реальный Polza model ID, endpoint и passthrough vibe **не подтверждены**; в плане это явно требует проверки в S01 ([W04], [W10], [W11]).
- Historical OpenRouter Gemini 3.1 (FAILED listening, один голос) — это legacy-маршрут, **не** доказательство поведения Polza Gemini 3.8.

## ASR

| Модель | Маршрут | Transcript/timestamps | Source-only | Live | Listening |
|---|---|---|---|---|---|
| Qwen local `Qwen/Qwen3-ASR-0.6B` | `qwen-local`: Python `qwen_asr` или настроенный audio.cpp | Python: `.text`, отдельный forced aligner для word; audio.cpp: `local_response.transcript` | Регистрация и оба mapping подтверждены кодом; загрузка моделей `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| Qwen local `Qwen/Qwen3-ASR-1.7B` | план-цель [W01] | неизвестно | **NOT registered** (registry знает только 0.6B); поддержка `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| Polza cloud `qwen/qwen3-asr-flash-2026-02-10` | план: POST `/audio/transcriptions` [W05][W06] | timing-поля и cost неизвестны | Adapter в registry **отсутствует**; контракт `BLOCKED_PROVIDER_CONTRACT` | `NOT_RUN` | `NOT_RUN` |

## Embeddings

| Backend | Модель | Маршрут | Что проверено | Размерность | Source-only | Live |
|---|---|---|---|---|---|---|
| local `local-qwen3-06b` | `Qwen/Qwen3-Embedding-0.6B` [W02] | Sentence Transformers (план) | адаптера в `src` и тестов backend в `tests` нет | 1024 — ожидание плана/upstream, **не** наблюдённое | План-цель, `NOT_IMPLEMENTED`; модель не загружалась | `NOT_RUN` |
| cloud `polza-qwen3-8b` | `qwen/qwen3-embedding-8b` [W14] | Polza POST `/embeddings` [W03] | адаптера в `src` и тестов backend в `tests` нет | 4096 — ожидание upstream, **не** наблюдённое | `BLOCKED_PROVIDER_CONTRACT` (shape/index/query/price не доказаны) | `NOT_RUN` |

## Внешние ссылки плана

`W01–W14` — существующие вторичные цитаты плана (официальные карточки Qwen/Google/Polza, SQLite, uv). Они **не** пере-выгружались в этой source-only сессии и сами по себе не доказывают реальный контракт Polza. Никакая живая проверка не выполнена.

## Минимальный будущий bounded-probe список (без выполнения)

1. Сначала уточнить реальные Polza ID/endpoint для обеих Gemini-моделей; после отдельного разрешения — по одному короткому live-запросу на модель для HTTP-shape и одного голоса. Offline dry-run этого не доказывает.
2. Одна короткая фраза на двух голосах + общий/частный vibe: проверить, что инструкция не зачитывается, и прослушать (listening).
3. Один `/audio/transcriptions` вызов на `qwen/qwen3-asr-flash-2026-02-10`: transcript, наличие/отсутствие timing-полей, cost.
4. Один `/embeddings` вызов `qwen/qwen3-embedding-8b`: `index`, длина/конечность векторов, фактическая размерность, цена.
5. Каждая проба — с явным approval, согласованными входами и лимитом числа запросов.

## Проверки отчёта

- Ссылка из `docs/README.md` существует; whitespace-проверки `git diff --check` и `git diff --no-index --check /dev/null docs/reports/2026-09-29-s01-source-only.md` — без диагностик.
- S01 HTTP-shape, аудио, ASR и vector tests: `NOT_RUN` (нет разрешённых live-запросов и новых адаптеров). Прежние offline pytest-гейты S02 не доказывают эти контракты.

## Residual uncertain claims (честно отдельно)

- Точное написание Polza-ID для Flash-Lite TTS неизвестно даже как candidate.
- Реальные голоса/лимиты/vibe для Gemini 3.8 через Polza — предположения из вторичных карточек, не подтверждены.
- Наличие timestamps у Polza Qwen ASR Flash не гарантировано ни планом, ни кодом.
- Фактические размерности и поведение `index` у обоих embedding-backend не наблюдались.
- Локальная 1.7B ASR и загрузка любой модели/векторов не проверялись.
