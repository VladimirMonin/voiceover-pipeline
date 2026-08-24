# Исследование: тяжёлые long-form hallucinations OmniVoice

Дата: 2026-08-24

## Статус и границы

Это research-only handoff для следующей карточки Default. Я не меняла `src/`, тесты, пользовательские контракты или skill, не запускала GPU/модель и не выполняла cloud/paid-вызовы. Единственный новый файл этой карточки — данный отчёт.

Владелец сообщил, что `voiceover-pipeline-retrospective-2026-08-24` имеет тяжёлый слышимый FAIL с большим количеством не относящейся к сценарию речи. Это человеческая оценка; технический decode/ffprobe PASS не является её заменой.

Вывод: **причина extensive hallucinated speech не установлена**. Имеющихся данных достаточно, чтобы отвергнуть объяснение «420 символов» или «whitespace padding» как самостоятельную причину. Они не достаточны, чтобы выбрать исправление или утверждать, что виноват конкретный design instruction, backend, quantization либо сам OmniVoice.

## Context7-first и upstream результат

Сначала выполнен Context7 resolve/query.

- `OmniVoice` разрешился как `/k2-fsa/omnivoice`; запросы long-form, conditioning и failure-modes вернули официальные страницы generation parameters и voice design.
- Прямой Context7 resolve для `audio.cpp` **не нашёл audio.cpp**: возвращены только кандидаты `Whisper.cpp` и `Llama.cpp`. Поэтому Query Documentation для audio.cpp не вызывался. Далее использованы официальный репозиторий `0xShug0/audio.cpp`, его OmniVoice-документ и исходник сессии.

Официальная документация OmniVoice говорит, что для long-form модель автоматически режет текст по целевой длительности аудио: default `audio_chunk_duration=15 s`, активация после estimated `audio_chunk_threshold=30 s`.[1] Она не устанавливает безопасный лимит в символах и не обещает сохранение текста при любой длине.

В voice-design документации `instruct` состоит из категорий speaker attributes. `very low pitch` — допустимый pitch attribute; `russian accent` заявлен только для английского synthesis text. Документация также предупреждает, что некоторые комбинации attributes могут срабатывать не так, как ожидается.[2] Поэтому Russian accent не является доказанным контролем русской речи и не должен смешиваться с low-pitch в одном эксперименте.

В upstream issue #116 maintainer связал отдельную проблему потери конечных фонем с chunk-wise inference, fade/pad и скоростью. Позже он указал на возможный дефект обучающих данных и отсутствие реального решения для этой проблемы.[3] Это подтверждает существование ограничений на границе чанков, но **не** доказывает причину посторонней галлюцинированной речи в этой карточке.

OmniVoice относит audio.cpp к community projects и прямо помечает такие проекты как не официально поддерживаемые OmniVoice team.[4] audio.cpp заявляет OmniVoice как TTS/Clone/Design/Ctrl family и portable backend surface для Windows, Linux и macOS, однако эти заявления не являются доказательством одинаковой quality/parity на конкретном Linux и Windows пути.[5]

В документации audio.cpp `--text-chunk-size` включает framework text chunking, а `audio_chunk_duration`/`audio_chunk_threshold` остаются model-side параметрами.[6] Его текущий исходник делает explicit text split при заданном `text_chunk_size`; без reference audio затем использует output первого чанка как reference для последующих и склеивает с fade/silence.[7] Это важная внешняя реализация, но архивные receipts фиксируют runtime revision `502b5b74…`, а не текущий upstream HEAD, поэтому нельзя переносить её поведение на historical run без отдельной parity-проверки.

## Репозиторный source map

Базовый репозиторный SHA в начале исследования: `cd1858dc9ebb66a3f22c05e95bc6899015d12a09`.

### Codebase

Полный индекс `home-v-code-voiceover-pipeline`: 3111 nodes, 10663 edges.

- Запрос `OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE` сузил owner до `config.py`, `local_tts_text.py`, `OmniVoiceLocalTTSProvider.synthesize_chunk` и `merge_omnivoice_session_fragments`.
- Запрос `runtime_sessions` нашёл `cli.generate`.
- Запрос `--text-chunk-size` сузил argv construction до `local_runtime/transports/audio_cpp_cli.build_audio_cpp_family_arguments`.
- Запрос `script_hash` сузил provenance semantics до `run_state.script_hash` и `run_state.initial_state`.

### Serena

Проверены точные symbol bodies и references:

- `local_tts_text.merge_omnivoice_session_fragments`: для non-dialogue OmniVoice объединяет внешние fragments в один runtime request. При `len(joined_text) > max_chars` добавляет только пробельные gutters, выравнивающие следующий boundary до fixed offset; spoken content не добавляется.
- `cli.generate`: после `prepare_local_tts_chunks` вызывает merge для `omnivoice-local` + admitted model и затем задаёт `runtime_session_count = len(chunks)`.
- `OmniVoiceLocalTTSProvider.synthesize_chunk`: всегда передаёт `text_chunk_size=OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE` в `LocalTTSRequest`; receipt проверяет WAV, но historical public receipt не включает transport argv, container identity или plaintext design instruction.
- `OmniVoiceLocalTTSProvider._mode_request_fields`: `design` передаёт `_design_instruction`; default fixed-style передаёт `OMNIVOICE_STYLE_CONDITION`.
- `config.OMNIVOICE_INTERNAL_TEXT_CHUNK_SIZE` сейчас имеет literal-значение `420`; `OMNIVOICE_STYLE_CONDITION` сейчас `female`.
- References `merge_omnivoice_session_fragments` подтверждают вызовы из `cli.generate` и tests, а не второй production request constructor.
- `run_state.script_hash` хеширует canonical JSON подготовленных `{id, text}` chunks, а не bytes исходного Markdown-файла. Следовательно receipt `script_hash` и SHA-256 текущего source file намеренно несравнимы как значения разных представлений.

### ast-grep

Структурный запрос `"--text-chunk-size"` нашёл все четыре production mode branches (`auto`, `clone`, `design`, fixed-style) в `src/voiceover_pipeline/local_runtime/transports/audio_cpp_cli.py`, а также две contract assertions. Ранее проверенный structural запрос `text_chunk_size=$VALUE` нашёл decoding transport payload в том же owner.

Текущая intent-only command shape для OmniVoice — `--task tts --family omnivoice --model … --backend cuda --text … --out …` плюс `--instruct … --text-chunk-size 420 --mode offline --seed 1234 --num-inference-steps 32 --guidance-scale 2.0` для design/fixed-style. Это **не** восстанавливает historical command: архивы не сохраняют original argv, execution source kind, container image или exact instruction.

## Локальные receipts и входы

Ниже «file SHA-256» означает SHA-256 current bytes на диске; «receipt script hash» — canonical prepared-chunks hash, описанный выше. Различие между ними не доказывает редактирование файла.

| Артефакт | Current input file bytes | Receipt script hash | Output, ffprobe | Внешние chunks / runtime sessions |
|---|---|---|---|---|
| Owner FAIL: voiceover-pipeline retrospective | 8,729 bytes; `3458a6ee09ad4ca5369d981ff76f3d3907a51e32cbbca61db20dd7c2c1a7d2cb` | `a2dc083288a5612e25236213432f8fbbbec1a29c9e9effb9eb64ebf83367b14f` | 296.544 s; 4,745,133 bytes | 1 / 1 |
| Better-by-owner: MDRack retrospective | 7,699 bytes; `0bc7faf3a76a5f52dfd99f65b5d71de85a9cfb0614a8a160e44f8dc3ef2d88ea` | `379b01cd7c01c0192520dd7838e9734f46b12ce7f8bce27b03f3bee150b8855a` | 258.312 s; 4,133,421 bytes | 1 / 1 |
| WVM accepted assembly | 15 retained input parts, total 5,112 bytes; per-part identities below | 15 per-run canonical hashes | final 181.128 s; 2,898,477 bytes | 15 / 15 |

The first two receipts agree on the same admitted model artifact `audio-cpp/omnivoice-q8_0`, model SHA-256 `2f4be637…3da8b`, Q8_0 GGUF, and runtime provenance `audio.cpp@502b5b74…`; both are Linux-side evidence (`/usr/bin/ffmpeg` and `/usr/bin/ffprobe` in receipts). Neither archived run public receipt preserves the actual design instruction, exact command, execution source kind, or container image digest.

A content search over every supplied failed/MDRack/WVM artifact tree found zero occurrences of `voiceover generate`; generation logs contain lifecycle events only. Thus an exact historical command cannot be recovered from these artifacts. Do not infer it from the current source route.

### Failed and MDRack one-session inputs

| Property | Owner FAIL | MDRack better-by-owner |
|---|---:|---:|
| Prepared runtime text characters | 5,336 | 5,817 |
| `internal_text_chunk_size` | 420 | 420 |
| `voice_session.strategy` | design-instruction-native-session | design-instruction-native-session |
| Seed | 1234 | 1234 |
| Generated whitespace padding runs | 12 | 13 |
| Generated whitespace padding total / maximum | 626 / 141 | 1,726 / 168 |
| Historical design identity digest | `design:71b4a877…` | `design:c6c169d0…` |

The two design identity digests differ. Plaintext instructions are intentionally absent: current transport code treats design instruction/text as sensitive argv content and forbids it in logs, receipts and metadata. No report should attempt to reconstruct it from the digest.

This is the decisive negative comparison: MDRack is longer and has more synthetic whitespace padding than the failed run, while the owner considers MDRack better. Therefore a fixed 420-character model split or whitespace gutter cannot alone explain the severe failure.

### WVM layout and quality evidence

The accepted WVM final was assembled from 15 listed `part-01`…`part-15` MP3s with 14 references to `assembly/pause.mp3`; the pause asset is 0.408 s. A stale `part-13b` generation directory exists but is not selected by `assembly/concat.txt`.

All active WVM run receipts use the same MDRack design identity (`design:c6c169d0…`), admitted model/runtime provenance, `internal_text_chunk_size=420`, fixed seed 1234, exactly one external chunk and one runtime session. Their prepared runtime texts range 102–312 characters, below 420; therefore the current merge owner would not insert its synthetic long-form whitespace gutter for any part.

| Part | Current bytes | Current file SHA-256 | Receipt canonical script hash |
|---|---:|---|---|
| part-01 | 220 | `c0439a0942932f802ccb3db527e9d619d09d1f37e3926914a0a289fa68cd71b4` | `17790f1f3b0f0bdde6eed6b07992048bed8fe257b1b2ccbffe2bb25a5c6e3ae5` |
| part-02 | 284 | `cc726d998f7498779a8d234bd49a9bde9ae8da01ffa74032e5787379dbbcbcee` | `25fd778da598b1eed496edc6d8e115ed53250bad60ed68ae72b91275d2fa5898` |
| part-03 | 196 | `11e7c48259c44328ef5f8dc6843a62bf85d97cf8456ed879100d108e5229d15e` | `650600d2a41137df0e28c4da333c210da297d758b0ffa2ed0084123daff16161` |
| part-04 | 395 | `9b269dc465a1b4ff80f6dcd571cf90f64498a766d768433445c0f939908d64f0` | `771152ba82989941c3363710c30addb06a60197316f21c069a1904ba1b249378` |
| part-05 | 189 | `e343fec555ccc26f5d8d88890a63b73a3b7f7d8ca3b2f36e2e5396591e4c5c8b` | `3c475130fc6b8b5c589408d5de2363346f196068a992088a1e22dfad4e168043` |
| part-06 | 524 | `23ea550bf71a9df381fdf8fe16c8d1ef159cf212fd88c177b2d7e7c3be357177` | `f3b8d11077a40b5439c07d6b84270b1f4f56bf51be0de7fc3b3ebaa292dd71e5` |
| part-07 | 343 | `c5f45a82c98f3c9c57c96c83ac29e8c378fdc24e9a1b81394aee686855f198a5` | `c965c78e25145e0560889b2189543789249d332eeaa883f928303e4f50e974c7` |
| part-08 | 385 | `6608b8ddad2146304793728db59eb5e22833a13b3e347ed213580684d1fa8ec8` | `8f91e76f1ae5a90eb4b54776c111b78b9b48281a46a6c4df4fb41f0212c9d36a` |
| part-09 | 479 | `b828b5d6db5ed577a758e023a55b8b64ac93781694aeaf3de0238cdb4d264115` | `49d166b1b9b48c95673e1447ca98086c31f713caefa2162db29be7f0a61159f1` |
| part-10 | 337 | `197dcdc7d138472253c9da93e16df90eb96a40e884d685bc8576ca6b1425b78e` | `6ad75400977385997d32185107728c8d6d0eb6e6d49ffa06f6a7450d28200af7` |
| part-11 | 420 | `fdb88d63dbf81275d04133be703001ffc601184520e02eb07a021f5e4f1c0566` | `022a618a6ed5723cab632153a0e563aa48c97250bb06d99678a6b50f078604a6` |
| part-12 | 207 | `4fd62d1a09809b6d5a9cc40c2386285f1532767738bc3f5285b7179f96ef44e5` | `790d7210e44fe7f53f227d7487193413dc3aea26cc0d6d639c77ddc89983a894` |
| part-13 | 266 | `05e5c9b7d03be779469d65acf5d2b190aacff1e4907ecb02b233ad3dbc7ef756` | `604a90b46c129bfd64fbbb1ed7de46e0bd2e9b24ba517db051669eb082809bb8` |
| part-14 | 286 | `7c401e2beaa6535671da50606dba8eb719ab18c69a5a4a65b8b84bc6b235ea5b` | `ff3fbbb6dcf649582ce59c3e198fccf176feaf7acc0fb2f6b1b8487b8581dbe2` |
| part-15 | 581 | `ad700b3547961e56097d070b4c0111f621784a2ae8b6a239daab6b780734bd03` | `d076c7fd62a5f4f136a6dcd1111e6b9450a53e4256627ed7397f58bad5d7fef3` |

WVM has an actual local-ASR receipt, unlike the failed and MDRack long single-session files. Final verification reports 361 expected vs 359 actual words, similarity 0.99444, and four token diff entries. Per-part verification reports 15 parts: 13 exactly 1.0, part-11 0.98039 and part-14 0.95. This is strong but imperfect machine evidence; it neither proves exact spoken fidelity nor replaces a human listening verdict.

## Historical 350-character evidence

The prior Terra run `t_33d50d9f` is the requested prior-run evidence, not a new claim from this card. It recorded one 509.85 s private WAV from one direct audio.cpp invocation over 8,147 spoken characters with `--text-chunk-size 350`: 23 hard seams, 18 within-word seams before the sentence-gutter change. Its local-ASR alignment did not establish a systematic one-letter truncation. The later sentence-safe preparation had 22 sentence-ending seams, zero word-splitting seams, but its single authorized post-fix synthesis exited 139 before writing a WAV. It therefore provides no audible after-fix validation.

This history supports testing 350 and 420 as independent factors. It does **not** establish that 350 is safe for long-form speech, and it does not establish a cause for the current severe hallucination.

## Causal assessment

### Proven

1. The failed file is a one-session, 5,336-character prepared text passed with internal limit 420 and 626 generated padding characters.
2. MDRack uses the same model artifact/runtime revision/seed/limit and is a longer one-session prepared text with more padding. The owner rates it better.
3. WVM uses a different execution layout: 15 separate, sub-420 prepared requests; it has an ASR receipt but not a perfect one.
4. Current source merges external non-dialogue OmniVoice fragments into one runtime request and forwards the 420 value to audio.cpp for every mode.
5. The exact historical command, actual design instruction, execution source kind, container digest and Windows route are absent from the supplied receipts. Linux is evidenced; Windows is not.

### Falsified explanations

- **"420 characters alone caused the severe failure"**: inconsistent with the longer MDRack single-session run at the same limit.
- **"whitespace padding alone caused the severe failure"**: inconsistent with MDRack having 1,726 generated padding characters versus 626 in the failure.
- **"decode/silence PASS proves quality"**: contradicted by the owner’s audible FAIL and by the absence of an ASR quality receipt for the failing output.

### Live hypotheses, not findings

- H1: an interaction between one unrecorded design instruction/voice identity and the long internal-chunk path causes degradation. It is confounded by different source text and no failing ASR transcript.
- H2: a particular content/token/punctuation region in the failed text triggers model/runtime degradation. Existing raw source bytes and receipt prepared text hashes preserve an investigation anchor, but no source-to-ASR alignment exists for this file.
- H3: the pinned audio.cpp revision has a long-form parity defect distinct from upstream Python. The official source describes splitting/reference propagation but no historical parity result is supplied.
- H4: a model limitation is involved. Upstream #116 supports an unresolved boundary omission limitation, not extensive invented speech; treating it as the explanation would overreach.

## Bounded next experiment for Default (not executed)

Do not run this matrix until the exact historical design instruction and execution route are recoverable from an owner-approved private source. Without them, the requested same-text/same-instruction A/B is not reproducible.

For one short adversarial source excerpt selected from the failed prepared text, hold model SHA, runtime revision, seed 1234, steps 32, guidance 2.0, backend, exact instruction and text constant. Run sequentially only after GPU preflight:

| Factor | Levels | What it can falsify |
|---|---|---|
| Internal chunk size | 350; 420 | A size-dependent degradation under identical conditioning |
| External layout | one long session; safe sub-420 external requests | Model-side internal path versus outer session layout |
| Padding | original fixed-offset gutters; no synthetic padding while preserving sentence text | Gutter/offset interaction without changing spoken text |
| Conditioning | exact recovered historical instruction; `female` only; one modifier at a time (`russian accent` or `very low pitch`) | Instruction interaction; never compare a combined change |

Every candidate needs a configured local-ASR measurement against the exact normalized expected text: major omission, unexpected insertion/hallucinated spans and repeated n-grams must fail closed. Preserve a privacy-safe receipt of ASR-route identity, expected-text digest, normalized metrics and audio digest. Keep human audible acceptance independent: an ASR PASS is not a human voice-quality PASS.

If the exact historical instruction/route cannot be recovered, classify the historical severe failure as **unreproducible provider/runtime-limited evidence**, not as an evidence-backed product bug. The smallest safe product response would then be a fail-closed long-form quality gate and explicit human acceptance requirement, but no implementation is proposed or made by this research card.

## Default follow-up: recovered command, bounded A/B, and product decision

Session history recovered the exact historical commands that public receipts
intentionally omitted:

- audible FAIL: `female, middle-aged, low pitch, russian accent`;
- owner-better MDRack: `female, middle-aged, very low pitch`.

Both used the same editable checkout command route, admitted model/runtime,
Russian text language, seed, steps and guidance already recorded above. This
removes the earlier uncertainty about the instruction, but it does not by itself
prove that the accent item caused the long-file hallucination because the source
texts differ.

Default then ran a bounded sequential same-text A/B on one 1,197-character
Russian excerpt at tracked baseline `cd1858dc9ebb66a3f22c05e95bc6899015d12a09`.
The only conditioning difference was the exact two instructions above. GPU
preflight showed 11,726 MiB free and zero compute applications; no cloud route
or concurrent model was used. Both files decoded, each was 68.016 seconds and
1,088,685 bytes, but their SHA-256 values differed (`d886a9d9…6ff3` versus
`f7fa2567…93ff`). Equal duration/size is not speech-quality parity.

The configured offline Qwen ASR route was attempted fail-closed with network
disabled, but its local model directory lacked model weights, so no transcript
or quality verdict was produced. The A/B therefore remains technical execution
evidence only and does not establish audible causality.

The evidence-backed product defect is narrower and certain: VOP accepted
`russian accent` for Russian synthesis even though upstream defines accent
attributes for English speech and warns that voice design is trained on Chinese
and English and can be unstable in other languages. VOP rejects English accent
attributes outside English and Chinese dialect attributes outside Chinese before
provider/runtime construction. The later upstream verdict also makes the
long-form nonclaim explicit: `female, middle-aged, very low pitch` is only a
syntactically valid historical instruction, not a solution to long Russian
hallucination. Long Russian design is an unsupported model regime; accepted
Windows clone/preset/voice-bank paths remain separate and are not invalidated.

The product also now writes path-free package execution identity to run receipts
and provides `voiceover verify-tts`: a configured local ASR compares expected
text with generated speech and fails on major omissions, unexpected insertions,
or repeated n-grams. Its durable receipt contains hashes/counts and ASR identity,
not source text or transcript; technical PASS explicitly retains a human
listening requirement.

## Non-claims

- The Terra research phase made no model run or patch; the later Default phase
  ran only the bounded Linux A/B above and did not listen to or semantically
  accept either file.
- Historical argv and design instructions were recovered from session history;
  the historical container image digest and Windows parity for the failed file
  remain unavailable.
- No statement here upgrades MDRack or WVM to a human audible acceptance verdict.
- The report does not claim that an upstream boundary-phoneme issue explains unrelated hallucinated speech.

## Sources

[1] https://github.com/k2-fsa/OmniVoice/blob/master/docs/generation-parameters.md — OmniVoice generation parameters
[2] https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md — OmniVoice voice design
[3] https://github.com/k2-fsa/OmniVoice/issues/116 — OmniVoice issue 116
[4] https://github.com/k2-fsa/OmniVoice/blob/master/docs/community-projects.md — OmniVoice community projects
[5] https://github.com/0xShug0/audio.cpp — audio.cpp repository README
[6] https://github.com/0xShug0/audio.cpp/blob/main/docs/models/omnivoice.md — audio.cpp OmniVoice model documentation
[7] https://github.com/0xShug0/audio.cpp/blob/main/src/models/omnivoice/session.cpp — audio.cpp OmniVoice session source
