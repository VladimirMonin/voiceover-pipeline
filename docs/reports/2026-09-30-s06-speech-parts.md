# S06 — speech-parts, короткая реплика, vibe и `--audio-format`

| Поле | Значение |
|---|---|
| Дата | 2026-09-30 |
| Ветка / HEAD | `main`, `b052fdc52f48e7f8c80456b27ed0e75a81997a97` — `feat: close S05 DB-first execution offline` |
| Изменённые файлы | 15 изменённых + 4 новых (`.pi/` и `.serena/` — user-owned, не тронуты) |
| Характер отчёта | **OFFLINE**: без сети, live-вызовов, прослушивания, платных запросов, загрузок моделей и чтения секретов |
| Статус | **`ACCEPT_OFFLINE_CORE / BLOCKED_PROVIDER_CONTRACT`**: offline-ядро принято родителем после исправления двух Sol6 P1, **полный S06 не принят**; Polza Gemini live/listening — `NOT_RUN` |
| Тестовые гейты | `pytest` **2080 passed / 2 skipped**; `ruff check`, `ruff format --check` (163 файла), `mypy --no-incremental` (86 source files), `git diff --check` — PASS на исправленных байтах |

Sol6 whole-stage review `99558a28-0732-4e3d-892b-5e4d8f0bc9c2` выдал **BLOCK**
на предыдущем кандидате: `--vibe` с обычным Markdown молча терялся и табуляция
после пробелов в отступе YAML проходила parser. Родитель добавил два fail-first
теста (до исправления: **2 FAILED**), запретил legacy `--vibe` до key/POST и
проверяет все символы отступа до разбора YAML. На исправленных байтах — 2 PASS,
156 focused, 2080 full passed / 2 skipped и Ruff/mypy/diff PASS. **Sol6 повторно
не пересматривал исправленные байты**; принятие offline-ядра — решение родителя,
не приписанный ревьюеру вердикт. Внешний Gemini-контракт остаётся блокером S06.

> Граница доказательств: реализовано и доказано офлайн всё, что не требует
> подтверждённого внешнего контракта: формат `speech-parts`, короткая реплика
> `--text/--vibe`, композиция `effective_vibe`, бюджет до первого POST, DB-first
> снимок и script-free resume, реальный WAV-контейнер итогового файла и
> fail-closed отказ для неподтверждённого Gemini-маршрута. Реальный контракт
> Polza Gemini 3.8 (id/endpoint/поле инструкции/голоса/контейнер/цена) **не
> подтверждён**: он остаётся `BLOCKED_PROVIDER_CONTRACT`, ни live, ни listening
> не выполнялись.

## Прочитанные источники истины

- `AGENTS.md` и routed-инструкции: `core`, `code-intelligence`, `agent-kanban`,
  `git-release-safety`, `provider-cli`, `test-quality`, `docs-governance`;
  родитель также применил `instruction-authoring`, обновив durable правило
  `--vibe` в `instructions/provider-cli.instructions.md`.
- План `docs/plans/2026-09-29-voiceover-pipeline-development-plan.md`
  §«Короткая реплика»/«Составная запись»/«Текст и инструкции разделены»/
  «Проверка длины без чанкера» (~214–312), S06 (~975–991).
- `docs/reports/2026-09-29-s01-source-only.md` — единственный источник по
  статусу Polza Gemini 3.8 (candidate ID; Flash-Lite ID неизвестен).
- Код: `services/prepare.py`, `services/native_generation.py`,
  `services/synthesis.py`, `history/native_snapshot.py`, `history/native_view.py`,
  `history/native_resume.py`, `artifacts.py`, `media.py`, `cli.py`,
  `providers/polza_tts.py`, `providers/openrouter_tts.py`.

## Ограничение инструментов (честно)

В этой worker-сессии **Codebase/Serena/ast-grep недоступны** (в наборе инструментов
только файловые/шелл-операции). Поэтому исследование опиралось на явно названные
родителем точки входа (`prepare.resolve_script_format`, `prepare_script_fragments`,
`cli._resolve_script_format`, `cli._native_route_eligible`,
`polza_tts._synthesize_audio_speech`, `voiceover_script.validate_model`,
`openrouter_tts._uses_documented_gemini_speech_contract`) и точный `grep` по ним.
AST-паттерн `is_dialogue_format($FMT)` подтверждён обычным поиском в `cli.py` и не
использовался как отдельный ast-grep-артефакт. Это зафиксированный gap, а не
скрытая замена.

## Что реализовано

### Формат и короткая реплика

- Новый модуль `speech_parts.py`: строгий загрузчик подмножества YAML (`version:
  1`, `format: speech-parts`, непустой `parts`, ровно одна непустая `voice` и
  непустой `text` на часть, опциональный общий и частный `vibe`). Отклоняются
  дубликаты ключей, неизвестные ключи, вложенные блоки и tab-отступы; BOM
  (`utf-8-sig`) принимается. `provider`/`model` в документе запрещены как
  неизвестные ключи.
- `compose_effective_vibe`: общий vibe, затем пустая строка, затем vibe части;
  отсутствующие значения пропускаются. `text` в инструкцию не попадает.
- `speech_part_request_chars(text, effective_vibe, required_text_wrapper)` —
  точный счётчик; лимит `SPEECH_PARTS_REQUEST_CHAR_LIMIT = 5000` проверяется для
  **всех** частей в `cli._preflight_speech_parts_route` до key/POST. Поздняя
  over-limit часть → JSON `SPEECH_PART_TOO_LONG`, exit `2`, ноль запросов.
- `generate --text ... --voice ... --vibe ...` создаёт ровно одну часть; вызов без
  `--text` сохраняет прежний default script; явные `--text`+`--script`
  взаимоисключающие; для `--format speech-parts` CLI `--voice`/`--vibe`/
  `--style-prompt` отклоняются (нет скрытых override). На обычном Markdown/
  dialogue/voiceover `--vibe` теперь **отклоняется** с
  `VIBE_UNSUPPORTED_FORMAT` (exit `2`) до чтения ключа/провайдера вместо
  молчаливой потери направления.
- `validate --format speech-parts --json` отдаёт `parts`, `request_chars`,
  `route.admitted`, `route.reason`; синтаксис/бюджет → exit `2`, неподтверждённый
  маршрут → warning `BLOCKED_PROVIDER_CONTRACT` при валидном документе.

### DB-first снимок и resume

- `PreparedPart`/`PreparedRun` получили `SpeechPartDirection` (`shared_vibe`,
  `specific_vibe`, `effective_vibe`) через `prepare_run(part_directions=...)`.
- `native_snapshot` пишет `vibe_shared`/`vibe_specific`/`vibe_effective` в
  уже зарезервированные колонки `parts` и в per-part fingerprint; ключи
  добавляются только при наличии, чтобы прежние прогоны сохранили байт-в-байт
  identity. Per-part direction разрешён только для `format: speech-parts`.
- `native_view` восстанавливает direction из config entry и сверяет его с
  committed-колонками (fail-closed при расхождении); `native_resume` сравнивает
  fingerprint с direction; `_reconstruct_prepared_run` несёт direction.
- Script-free resume: снимок несёт точный текст частей, `history resume` не
  перечитывает исходный файл; для `--text` (нет файла) `script_path=None`
  разрешён только для `speech-parts` (остальные маршруты по-прежнему fail-closed).
- `speech-parts` исполняется только нативным исполнителем; legacy fallback
  отсутствует (`NATIVE_OPTIONS_UNSUPPORTED` при неподходящей смеси опций).

### Исполнение части и vibe

- `services/synthesis.synthesize_part` передаёт провайдеру `voice`, если callable
  его принимает, и `vibe` (точный `effective_vibe`), если callable принимает
  `vibe`; при непустом vibe и отсутствии `vibe`-параметра вызов **fail-closed до
  POST** (никогда не молчит).
- Нативный исполнитель для `speech-parts` идёт через тот же `synthesize_part`,
  что и dialogue-turn; порядок = порядок `parts`, голос на часть.

### `--audio-format`

- `--audio-format {mp3,wav}` (default `mp3`) задаёт контейнер итогового merged
  файла: `build_run_paths(..., audio_format=...)` меняет расширение, `media`
  concat/`concat_dialogue_turns` для `.wav` используют `pcm_s16le` (реальный
  RIFF/WAVE), запись history mime — `audio/wav`. Промежуточные файлы в `chunks/`
  остаются MP3. Формат записывается в output options только при не-default
  значении, поэтому resume прежних MP3-прогонов не инвалидируется, а смена
  формата на `--resume` отклоняется (`_ERROR_PROCESSING_CHANGED`). `wav`
  допускается только на нативном маршруте.

### Provider admission (inert candidate)

- Новый `speech_parts_route.py`: candidate `polza-tts/google/gemini-3.8-flash-tts`
  (`verified=False`), `required_text_wrapper=""` (не наблюдался).
  `require_confirmed_speech_parts_route` отказывает с `BLOCKED_PROVIDER_CONTRACT`
  (exit `30`) при кандидатной модели, при любом непустом vibe (ни один
  подтверждённый маршрут не переносит инструкцию, не зачитывая её), и при
  различающихся голосах частей вне подтверждённого per-part-voice маршрута. Ключ
  и POST не выполняются. Flash-Lite ID не выдуман и не регистрируется.

## Файлы

Изменено 15 (код/док/владеющая инструкция), добавлено 4 (два модуля,
тесты и этот отчёт).

| Файл | git blob |
|---|---|
| `src/voiceover_pipeline/speech_parts.py` (new) | `8a9d2f847b35404747f2da0df897efbeb64029ad` |
| `src/voiceover_pipeline/speech_parts_route.py` (new) | `baca71f0ae02eb77ecafe8a698b819a714841e60` |
| `tests/test_speech_parts.py` (new) | `337f8ab2f5ef34a471f76c7c5194297117e11ae2` |
| `src/voiceover_pipeline/cli.py` | `886488b4d79a85c825d3edea371c47e77814a503` |
| `src/voiceover_pipeline/services/prepare.py` | `0761ecf9e518558f9cfcddb3dcd966ff35ea22cc` |
| `src/voiceover_pipeline/services/synthesis.py` | `67f53be982044b1c4cfbffea23c6f0562f305c98` |
| `src/voiceover_pipeline/services/native_generation.py` | `e4b9ae80376031f382ee13f2b810ea7179bfc80c` |
| `src/voiceover_pipeline/history/native_snapshot.py` | `789a6ebbbec9ffabc29ef1bfe744e3ffdf5db0f5` |
| `src/voiceover_pipeline/history/native_view.py` | `990a2ba6f277ecc3fadd15fffb4e1178c1b5b51c` |
| `src/voiceover_pipeline/history/native_resume.py` | `1554c8a11d9e519616f071aff22e2e32480cab22` |
| `src/voiceover_pipeline/history/native_export.py` | `5bd4b8078c862d910743e128f46e68466f020815` |
| `src/voiceover_pipeline/artifacts.py` | `175df307e9d8ed2b593dca80ea180d77c515f5d7` |
| `src/voiceover_pipeline/media.py` | `a177f0f0a388a2afd2ef67f11b3e64c7665139b5` |
| `src/voiceover_pipeline/run_state.py` | `809b15f930f22bfcf9fc9a45ed2555a9f2fed8e7` |
| `docs/agent-cli-contract.md` | `d8382632c110d1ba2717a5c9b64509a94f4c6872` |
| `docs/skills/voiceover-pipeline/SKILL.md` | `d301d9d4595b457941e5c48cc188293a70588a69` |
| `docs/README.md` | `5df5e94bade0b0a4ebabdbe2c9209c9b751a9ee5` |
| `instructions/provider-cli.instructions.md` | `37e2ae42e9fd2f18db83c58bed1519668fc8cb0c` |
| `docs/reports/2026-09-30-s06-speech-parts.md` (этот отчёт, new) | — |

## Тесты

`tests/test_speech_parts.py` (28 кейсов после Sol6 P1 fixes):

- парсер: BOM, tab после пробелов в отступе, дубликаты/неизвестные ключи,
  version/format, пустые `parts`,
  пустой `voice`/`text`, отсутствующий vibe-эффект, композиция обоих vibe;
- бюджет L-1/L/L+1 включая emoji и непустой `required_text_wrapper`;
- `validate`: blocked candidate route и синтаксическая ошибка (exit `2`);
- CLI: взаимоисключение `--text`/`--script`, отклонение `--voice` для
  speech-parts и `--vibe` для legacy script, ранний `BLOCKED_PROVIDER_CONTRACT`
  для vibe и candidate-модели (провайдер/ключ не строятся), поздняя over-limit
  часть → ноль POST;
- native e2e: 3 части / два голоса / разные vibe, порядок, голоса, отсутствие
  инструкции в произносимом тексте, колонки `vibe_*` и JSON `script_format`;
  **тест подменяет только admission policy на fake confirmed provider** — это
  проверка внутренней передачи данных, а не реально доступный paid-маршрут;
  в production непустой vibe по-прежнему блокируется до подтверждения Polza-контракта;
- script-free resume (файл удалён) и `--text`-прогон без script path;
- отказ resume при смене `--audio-format`;
- реальный WAV: `media.concat_audio_files` с настоящим `ffmpeg` даёт RIFF/WAVE
  (skip без ffmpeg) + CLI-прогон `--audio-format wav` с `.wav`-расширением и
  записанной опцией.

Команды:

```
uv run --offline --frozen pytest -q -p no:cacheprovider tests/test_speech_parts.py tests/test_cli_json_contract.py tests/test_history_native_resume.py tests/test_history_native_voiceover.py  # 156 passed
uv run --offline --frozen pytest -q -rs -p no:cacheprovider  # 2080 passed, 2 skipped
uv run --offline --frozen ruff check src tests              # PASS
uv run --offline --frozen ruff format --check src tests     # 163 files formatted
uv run --offline --frozen mypy --no-incremental             # 86 source files, PASS
git diff --check                                             # PASS
git diff --no-index --check /dev/null <new file>  # diagnostics empty
```

## Остающиеся вопросы по контракту провайдера (не выполнялись)

1. Точный Polza model id и endpoint для Gemini 3.8 speech-parts; id Flash-Lite.
2. Поле запроса, переносящее инструкцию так, чтобы она **не** зачитывалась;
   точный `required_text_wrapper` и `request_chars`-маппинг.
3. Список голосов на часть; допустимость двух разных голосов в одном прогоне.
4. Аудио-контейнер ответа и поддержка `wav`/`mp3` в `/audio/speech`.
5. `usage`/cost и граница длины `input`/инструкции для этой модели.

Ограниченный запрос на approval (для владельца, **не исполнялся**): одна короткая
проба на модель, 2 голоса и один общий+частный vibe — проверить HTTP-shape,
отсутствие зачитывания и прослушать; число запросов — не более 2, входы —
согласованные, бюджет — минимальный. До этого маршрут остаётся
`BLOCKED_PROVIDER_CONTRACT`.

## Честный residual

- `live`/`listening` для Gemini-маршрута — `NOT_RUN`; модель не
  зарегистрирована как stable и не рекламируется.
- `--audio-format wav` меняет контейнер только итогового merged-файла; части в
  `chunks/` — внутренние MP3 (совместимость с командой `concat`).
- Codebase/Serena/ast-grep в сессии недоступны — зафиксированный gap.
