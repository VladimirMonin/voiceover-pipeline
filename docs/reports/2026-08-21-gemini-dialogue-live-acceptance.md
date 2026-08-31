# Приёмка: live Gemini dialogue — двухспикерный smoke-прогон (0.6.0)

Date: 2026-08-21

> **Status: FAILED/BLOCKED — 0.6.0 must not be tagged/published until two
> distinct voices are proven audibly.**
>
> Historical decision note: this report remains evidence for the failed legacy
> multi-speaker request. On 2026-08-24 the owner explicitly authorized PyPI
> publication before the replacement turn-by-turn route completes its final
> human listening gate. This does not convert the historical audio to PASS.
>
> Обновлено 2026-08-22 после прослушивания: audible cast assignment FAILED.
> OpenRouter проигнорировал недокументированное поле
> `multi_speaker_voice_config` и синтезировал весь диалог с одним голосом —
> женским `Kore` — для обоих спикеров. Транспорт и генерация аудио PASS;
> план фикса: [docs/plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md](../plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md).

Scope: одобренный платный live-прогон двухспикерного сценария
`gemini-dialogue` через OpenRouter: генерация, resume без смены каста,
отклонение resume после смены каста, отсутствие секретов в артефактах.

Прогон выполнен из временного каталога
`C:\Users\User\AppData\Local\Temp\opencode\gemini-smoke-20260821-225042\`;
аудиоартефакты лежат в gitignored `out/` внутри репозитория. Tree репозитория
не изменён, кроме данного отчёта.

## Окружение

- Windows native, Python 3.14.2, uv
- Провайдер: OpenRouter, модель `google/gemini-3.1-flash-tts-preview`
- Doctor: `openrouter_key` ok, ffmpeg/ffprobe ok; CUDA недоступен —
  только optional-предупреждение, на облачный TTS не влияет
- Дата: 2026-08-21

## Сценарий под тестом

- Формат `gemini-dialogue`, язык `ru`
- Голоса: Host = Kore (Ведущая, warm confident technical host),
  Guest = Puck (Гость, calm thoughtful technical expert)
- Стиль: русский технический подкаст, формат Q&A; разрешённые теги
  `warmly` / `curious` / `serious` / `short pause`
- `max_chunk_bytes` 3500; 2 чанка по 2 реплики; всего 305 UTF-8 байт
- Полный текст промпта в отчёт не включается

## Процедура

1. Оффлайн-валидация сценария (без сети): exit 0,
   `{"status":"success","valid":true,"chunks":2,
   "speaker_voice_map":{"Host":"Kore","Guest":"Puck"},"errors":[],"warnings":[]}`
2. Doctor (redacted): `{"status":"success","required_ok":true,"workflow_ok":true}`
3. Платная генерация (одобрено заранее): exit 0
4. Resume того же каста (без платного запроса): exit 0
5. Смена каста (Host Kore → Zephyr) на том же run-id
   (без платного запроса): exit 30

## Результаты приёмки

| # | Проверка | Результат | Доказательство |
|---|---|---|---|
| 1 | Оффлайн-валидация сценария | PASS | exit 0, valid=true, chunks=2, пустые errors/warnings |
| 2 | Doctor: обязательные проверки | PASS | required_ok=true, workflow_ok=true; CUDA — optional-предупреждение |
| 3 | Платная генерация | PASS | exit 0, run_id `smoke-20260821-225042`, длительность 14.16 s (подтверждено ffprobe) |
| 4 | Назначение голосов спикерам | PASS (transport) | `speaker_voice_map` + стилевой промпт переданы по каждому запросу (request path) |
| 5 | Отсутствие подмены ролей | PASS (transport) | Host→Kore, Guest→Puck во всех запросах, без свапа |
| 6 | MP3 и манифесты существуют | PASS | full + 2 чанка MP3, run/manifest/chunks/run_state JSON — хеши ниже |
| 7 | `--json` — один объект | PASS | во всех трёх прогонах (validate / generate / resume) |
| 8 | Resume без смены каста | PASS | exit 0, resume_detected completed=2, chunk_skipped_resume ×2, без provider-вызовов; хеши артефактов идентичны |
| 9 | Resume после смены каста | PASS | exit 30, `{"status":"error","error":"Cannot resume: voice identity changed.","code":30}`; log `resume_rejected reason=voice_identity_mismatch`; платный запрос не выполнялся |
| 10 | Секреты в логах/артефактах | PASS | regex-скан всех текстовых артефактов чист: нет ключей и ссылок на `.env` |
| 11 | Аудируемость голосов (прослушивание) | **FAIL** | пользователь услышал один женский голос (`Kore`) для обоих спикеров; OpenRouter игнорирует `multi_speaker_voice_config` и применяет один top-level `voice` |

Дополнительно зафиксировано:

- `run_state.json`: `voice_identity` =
  `9da6acd18811ba953d5166d03b3d8b474f95efd0cfca810c3e1020fda2377110` (64 hex)
- `chunks.json`: `script_format` = `gemini-dialogue`,
  `speaker_voice_map` = `{Host: Kore, Guest: Puck}`, `prompt_mode` = `native`,
  стоимость по чанкам

## Артефакты (SHA-256)

| Артефакт | Размер | SHA-256 |
|---|---|---|
| full MP3 | 227757 B | `CB6DE7F15751D7AFE2A7C26CA4ACC41D57B68D419D8BD1CC47873A270323F865` |
| run JSON | 3563 B | `DD1A1C618EE29960BCC20FEC50377C6BDA4C73E904AE36FDC14CE1F489F7C688` |
| chunks.json | 3260 B | `7353B3E861734065B6EAB658C65560046934878B5D1727D172132DC8780E78CA` |
| manifest.json | 509 B | `824BD6F952500ECB3FDB36AB337697B624EF6235612A54DC493A72BB26864F20` |
| run_state.json | 2822 B | `26F62D2AF4F38375B6F026852DE4A11E630D223257F97CE8E625547C5B27365E` |
| generation.log | 1413 B | `A310BA5749EC70CE582302C7A7D4B6E1DC961C9EC09A7E82C25E2071F5496260` |
| chunk_01.mp3 | 100653 B | `0442E541654F3AD533D1458BD2E9DF32785EC119704B697E17090A7B88AACE16` |
| chunk_02.mp3 | 128685 B | `9DB1E1FF86481E2BE4F0ABB29EE333BF42E5D681CAE86055AA6ED6A886173DF21` |

## Стоимость

- Итого: USD 0.007142 (chunk 1 — 0.003131, chunk 2 — 0.004011)
- Ценовой snapshot на момент прогона: $0.000001 / $0.00002 за токен —
  snapshot, не гарантия провайдера
- Один короткий smoke-прогон; цена может меняться

## Открытые пункты

- Аудируемость голосов: **FAILED** — пользователь прослушал результат и
  услышал один женский голос (`Kore`) для обоих спикеров. Причина:
  OpenRouter `/api/v1/audio/speech` поддерживает только один top-level
  `voice` на запрос; поле `multi_speaker_voice_config` не документировано,
  провайдер его игнорирует. Транспорт (два запроса, карта голосов, resume
  identity) работал, но audible cast assignment не произошёл.

## Заключение

**Итоговый статус: FAILED/BLOCKED.** Транспорт и генерация аудио прошли
(PASS): платная генерация успешна (run_id `smoke-20260821-225042`, 2 чанка,
14.16 s, USD 0.007142), resume без смены каста не регенерирует, resume после
смены каста отклоняется до платного запроса, секреты не обнаружены. Но
**audible cast assignment FAILED**: пользователь услышал один женский голос
(Kore) для обоих спикеров, поскольку OpenRouter игнорирует недокументированный
`multi_speaker_voice_config`. Два различных голоса не доказаны на слух, поэтому
0.6.0 **не должен быть tagged/published** до тех пор, пока не будет
прослушана и принята по audibility двухголосая версия. План фикса (по одному
запросу на реплику с одним документированным top-level `voice`):
[docs/plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md](../plans/2026-08-22-agent-first-twovoice-dialogue-fix-plan.md).
