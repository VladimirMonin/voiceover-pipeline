# S02 — Деньги, неопределённый submit и оплаченный исходник

| Поле | Факт |
|---|---|
| Дата / ветка | 2026-09-29 / `main` |
| Принятый кодовый HEAD | `141a4d5424e7faaf85816c349a30c95072c0764e` — `fix: preserve paid raw audio before conversion` |
| Статус | Основные критерии S02 **ACCEPTED OFFLINE**; реальные provider-контракты, live и listening — **NOT_RUN**, см. пробелы ниже |
| Границы | Без paid/live/provider вызовов приложения, загрузки моделей, чтения `.env`, tag, release или push этого среза |

## Изменения и SHA

Ниже именно S02-срезы; `a710d85…` между Media и raw — отдельный S01 source-only отчёт, а не доказательство S02. Все SHA — локальные Git-объекты; первые два среза были pushed ранее, остальные и этот отчёт **не pushed**.

| Commit | Тема | Изменённые файлы |
|---|---|---|
| `f634cb84ea7a2d076aa227fe1908710c2512ea6c` | Нулевая цена остаётся наблюдённой | `src/voiceover_pipeline/{cli,pricing}.py`, `tests/{test_cli_json_contract,test_pricing}.py` |
| `1da1585708fef931885ceaadb1668d0cec5e5172` | Только подтверждённый generation ID связывает цену | `src/voiceover_pipeline/cli.py`, `tests/test_pricing.py`, `docs/{artifacts-and-analysis,polza-tts-models}.md` |
| `aa7ded3446c33d8646a65e15723633adb920c345` | Decimal-цена и сохранение при resume | `src/voiceover_pipeline/{cli,pricing,run_state}.py`, `src/voiceover_pipeline/providers/polza_tts.py`, `tests/{test_cli_json_contract,test_generation_stability,test_new_providers,test_pricing}.py`, `docs/{artifacts-and-analysis,polza-tts-models}.md` |
| `795482a39dd9e975c5c24f8954a03d66118acdd6` | Маркер до paid POST, нет автоматического resubmit/fallback | `src/voiceover_pipeline/{cli,run_state}.py`, `src/voiceover_pipeline/providers/polza_chat_audio.py`, `tests/{test_generation_stability,test_new_providers}.py`, `docs/agent-cli-contract.md`, `docs/polza-openai-audio-models.md`, `docs/skills/voiceover-pipeline/docs/{00-version-log,04-input-format,05-providers-and-models,06-commands-and-flags}.md`, `instructions/provider-cli.instructions.md` |
| `1ba574341c1dc7f627ada124a26310229ccf1160` | Известный Polza Media ID восстанавливается GET-only | `src/voiceover_pipeline/{cli,run_state}.py`, `src/voiceover_pipeline/providers/polza_tts.py`, `tests/{test_generation_stability,test_new_providers}.py`, `docs/agent-cli-contract.md`, `docs/skills/voiceover-pipeline/docs/{05-providers-and-models,06-commands-and-flags}.md`, `instructions/provider-cli.instructions.md` |
| `141a4d5424e7faaf85816c349a30c95072c0764e` | Raw до FFmpeg; локальный resume; WAV и стоимость direct speech | `src/voiceover_pipeline/{cli,media,run_state}.py`, `src/voiceover_pipeline/providers/polza_tts.py`, `tests/{test_generation_stability,test_new_providers}.py` |

Этот документ, индекс и синхронизация CLI/skill/instruction — отдельная последующая документационная правка; её commit не подменяет кодовый SHA.

## Доказанные офлайн-контракты

- `0` — наблюдённая стоимость, а не отсутствие цены. Канонические строки `Decimal` дают точную сумму (`0.1 + 0.2 = 0.3`); совместимые публичные float-поля остаются, но не служат доказательством точности. Polza chat-audio cost привязывается к подтверждённому generation ID; неподтверждённую Polza TTS history цену не подбираем по позиции.
- `pending_attempt` записывается **до** платного POST. Таймаут, сбой и неопределённый outcome не дают автоматического второго POST через retry, voice fallback, `--resume` или `--overwrite`. Блок `PAID_SUBMIT_UNCONFIRMED` наступает до key/provider/pricing. Сохранённый валидный Polza `/media` task ID разрешает лишь GET poll/download той же задачи и той же identity; exact cost после poll сохраняется до download.
- Принятые bytes любого paid TTS атомарно пишутся в run-local `raw/` до FFmpeg, затем bounded receipt (format/path/SHA-256/generation ID/известная цена) попадает в `run_state.json`. После ошибки FFmpeg `--resume` с тем же сценарием и целым raw файлом локально пересобирает незавершённую часть **без POST или Media GET**; raw не удаляется после успеха. Без пригодного raw повторный POST всё равно заблокирован; если есть валидный Media ID, допускается только GET-only восстановление. `status --json` согласован с проверкой raw/предшествующих MP3, но не знает аргументов будущей команды.
- RIFF/WAVE-байты получают WAV receipt даже при отсутствующем/ошибочном поддерживаемом MIME Polza `/audio/speech`; FFmpeg декодирует контейнер, а не копирует WAV в `.mp3` и не читает его как `s16le`. Неподдерживаемый MIME не угадывается как PCM; `audio/L16` отклонён, поскольку его big-endian семплы нельзя читать как little-endian `pcm16`. Exact direct speech cost записывается с raw receipt и переживает сбой конвертации/resume.
- Модели и голоса не заменялись. Удаление run через `--overwrite` при **любом** pending marker всё ещё закрыто; работоспособность реального провайдера этим не доказана.

## Проверки на принятых байтах

- Fail-first raw-тесты на HEAD до raw-кода: **8 failed, 5 passed**. При временном снятии только трёх review-fixes: форматная выборка **4 failed, 9 passed**, интеграционные WAV/стоимость **2 failed**; после восстановления fixes зелёные.
- Родительский `uv run --offline --frozen pytest -q tests/test_new_providers.py tests/test_generation_stability.py tests/test_cli_json_contract.py`: **238 passed**.
- Родительский `uv run --offline --frozen pytest -q -p no:cacheprovider`: **976 passed, 2 skipped**.
- `uv run --offline --frozen ruff check src tests`: PASS; `ruff format --check src tests`: **101 files**; `mypy --no-incremental`: PASS, **56 source files**; `git diff --check`: PASS.
- Перед scoped commit 6 файлов index был пуст; принятый `git diff --binary` SHA-256: `153e588d71d00e406935982d7c666571f5d48be7587f55ff2a98b262fa8387b7`. Независимый Sol6 сначала нашёл три P1 (WAV как MP3, потерю direct exact cost, L16 endian); повторная **fresh read-only** проверка после исправлений: **OK, no issues found**. Reviewer не мог сам запустить Git/гейты в своём инструментальном sandbox; указанные проверки — родительские, не его.
- `git status --short` после кодового commit: только пользовательские `?? .pi/` и `?? .serena/`; index чист. Ни один из них не включён в commit.

## Не доказано / дальнейшие границы

- **Live/listening NOT_RUN**: без разрешения не проверялись реальный Polza response shape/MIME/ID/стоимость, OpenRouter и chat-audio billing, звук, доступность моделей, цена и скорость. S01 source-only не заменяет эти проверки; S01 live/listening остаётся `BLOCKED_PROVIDER_CONTRACT` до решения владельца.
- Сеть и диск не могут быть атомарны вместе: сбой после оплаченного ответа, но до записи raw receipt может оставить заблокированный outcome без восстанавливаемых байтов. Не обещается восстановление при любом отказе.
- План S02 перечисляет имена лог-событий `retry_scope` и `submit_outcome_unknown`; точные имена здесь **не реализованы/не проверены**, хотя реальный paid retry подавлен и marker status `outcome_unknown` сохраняется. Общая redaction ошибок логгера — отдельный P1 плана; текущий отчёт не утверждает её завершение.
- Неподдерживаемые `audio/L16`, FLAC/OGG на текущем `/audio/speech` маршруте отвергаются вместо декодирования; их live-контракты неизвестны. Отдельного разрешённого повторного paid-submit или новой cleanup-команды этот срез не вводит. S03–S12 и полная S01-приёмка остаются отдельными этапами.
