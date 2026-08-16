# Архитектурная записка: гибридная консолидация VOP на audio.cpp

**Дата уточнения цели:** 2026-08-16
**Voiceover Pipeline:** commit `0dccdd4b0d89762c2379123fe8df60282147c547`
**Базовый аудит:** [Оценка применимости audio.cpp](2026-08-15-audio-cpp-feasibility.md)

Этот документ не отменяет прежний отказ от big-bang миграции. Он фиксирует уточнённую цель владельца: сохранить Faster-Whisper и облачные providers, а все остальные локальные ASR/TTS families оценить для поэтапной консолидации на общем audio.cpp runtime.


## Executive verdict

**Да — hybrid consolidation технически существенно разумнее прежнего full-migration framing.**

Новая цель не требует от audio.cpp заменить то, чего у него нет: Faster-Whisper и облачные providers. Она использует его сильную сторону — единый native/GGUF runtime для **Qwen3-ASR, Nemotron 3.5 ASR, Qwen3-TTS и OmniVoice** — при сохранении проверенных независимых маршрутов.

Итог:

- **ACCEPT SPIKE:** единый audio.cpp runtime для всех local non-Whisper families.
- **PROMOTE IF PASS:** каждое семейство отдельно, только после сравнительного runtime-gate.
- **REJECT:** big-bang, удаление Python adapters до доказанной паритетности, удаление Faster-Whisper/cloud.
- **DEFER:** production streaming и универсальный daemon до доказательства хотя бы одного ASR и одного TTS family.
- Нужны не только `AudioCppASRProvider`/`AudioCppTTSProvider`, а отдельный **`LocalAudioRuntime` — владелец процесса, GPU lease, unload, cancellation, provenance и privacy**.
- Текущий feasibility report нуждается в **датированном addendum/revision**: прежний вывод был корректен для полной миграции, но его рекомендация «optional ASR only; TTS deferred» больше не отвечает уточнённой цели и не учитывает OmniVoice.

---

## 1. Текущий inventory VOP

Проверен commit `0dccdd4b0d89762c2379123fe8df60282147c547`.

### Local routes

| Route | Текущая модель/runtime | Контракт |
|---|---|---|
| `qwen-local` ASR | `Qwen/Qwen3-ASR-0.6B`, Python `qwen-asr` | batch text; forced language; contextual text; без timestamps |
| `nemotron-local` ASR | `nvidia/nemotron-3.5-asr-streaming-0.6b`, Transformers/RNNT | текущий публичный route — batch text; без context/timestamps |
| `qwen-local` TTS preset | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | speaker, instruction; default Aiden, включая Sohee |
| `qwen-local` TTS clone | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | reference WAV; reference text optional, с x-vector-only fallback |
| `faster-whisper` | base/small/medium/large-v3-turbo/large-v3 | отдельный timing route, segments + words |

Cloud routes остаются отдельными: Polza/OpenRouter TTS и OpenRouter/Groq/xAI transcription/timings.

Ключевые anchors:

- `src/voiceover_pipeline/providers/qwen_asr_local.py`
- `src/voiceover_pipeline/providers/nemotron_asr_local.py`
- `src/voiceover_pipeline/providers/qwen_local.py`
- `src/voiceover_pipeline/providers/faster_whisper.py`
- `src/voiceover_pipeline/providers/asr_registry.py`
- `src/voiceover_pipeline/models.py`
- `src/voiceover_pipeline/cli.py`
- `src/voiceover_pipeline/config.py`
- `pyproject.toml`

Code-intelligence evidence:

- **Codebase:** project `home-v-code-voiceover-pipeline`, 1,485 nodes / 4,311 edges; owners — registry, adapters, CLI, benchmark.
- **Serena:** `QwenLocalASRProvider/transcribe`, `NemotronLocalASRProvider/transcribe`, `QwenLocalTTSProvider/synthesize_chunk`, `ASRRequest`, `ASRResult`, `ASRCapabilities`, `SynthesisResult`, `build_provider`.
- **ast-grep:** 4 `TTSProvider` implementations; ровно 2 production `ASRProviderSpec`; explicit CLI branches для четырёх TTS providers.

---

## 2. Что audio.cpp может заменить

На exact audio.cpp commit `502b5b74bd26e9b4aed267d1776ecf131cae7215` все четыре целевых семейства находятся в core registry: Qwen3-ASR, Nemotron ASR, Qwen3-TTS и OmniVoice.[1][2]

| Slice | Source support сейчас | Production replacement |
|---|---|---|
| Qwen3-ASR 0.6B | Есть offline/streaming, context, language; timestamps через отдельный aligner | Только после 50-case spike |
| Nemotron 3.5 ASR 0.6B | Есть offline/streaming, locale и word timestamps | Только после spike; не объявлять confidence без output proof |
| Qwen3-TTS Base | Есть cloning | После TTS parity spike |
| Qwen3-TTS CustomVoice | Есть named speaker + instruction | После Aiden/Sohee parity |
| Qwen3-TTS VoiceDesign | Есть, но сейчас такого VOP route нет | Новая capability, отдельный UX/contract slice |
| OmniVoice | Есть Auto/Clone/Design/control и pseudo-streaming | Новый provider после license + quality/resource gate |
| Faster-Whisper | Нет | Не заменять |
| Cloud providers | Не относятся к runtime | Не менять |

Факт наличия loader/session/package ещё не доказывает качество, VRAM-fit или стабильность на целевой GPU.

---

## 3. OmniVoice: три разных реализации

### 3.1 Official Python `k2-fsa/OmniVoice`

- Qwen3-0.6B-based diffusion TTS; **646 языков**, включая `ru` (Russian — строка 481 language list).[8][9]
- Voice cloning, voice design, auto voice, non-verbal tags, English/Chinese pronunciation controls, duration/speed и long-form chunking.[8][10]
- Clone: рекомендуемый reference clip **3–10 секунд**. `ref_text` можно не передавать — Python route загрузит Whisper и автоматически транскрибирует reference audio.
- Design не требует reference audio, но upstream прямо предупреждает: design обучался преимущественно на Chinese/English; для русского результат возможен, но стабильность не обещана.
- Output Python API: список mono `np.ndarray`, **24 kHz**; CLI пишет WAV.
- Official weights: `model.safetensors` ≈2.45 GB плюс audio-tokenizer ≈0.806 GB. Это размеры файлов, **не VRAM/RAM expectation**.
- Runtime RAM/VRAM для целевой машины не опубликованы; H100 benchmark нельзя переносить на локальную GPU.

### 3.2 Standalone `ServeurpersoCom/omnivoice.cpp`

- Отдельный C++17/GGML port, commit `4f33af825d66e6ef1cb185e87b4589cacf747291`.
- Заявляет 646 languages, CPU/CUDA/ROCm/Metal/Vulkan, WAV 24 kHz mono, clone и attribute-based design.[12]
- Clone требует **reference WAV плюс transcript**; умеет заранее кодировать reference в `.rvq`.
- GGUF состоит из base + tokenizer:
  - F32 ≈3.19 GB суммарно;
  - BF16 ≈1.60 GB;
  - Q8_0 ≈0.945 GB;
  - Q4_K_M ≈0.660 GB.[13]
- Это полезный независимый parity oracle, но **не часть audio.cpp и не доказательство его совместимости или производительности**.

### 3.3 `audio.cpp` family `omnivoice`

Exact anchors:

- `model_specs_v1/omnivoice.json`
- `src/models/omnivoice/loader.cpp`
- `src/models/omnivoice/session.cpp`
- `src/models/omnivoice/prompt_builder.cpp`
- `src/models/omnivoice/language_map.inc`
- `docs/models/omnivoice.md`

Свойства:

- README: `TTS, Clone, Design, Ctrl`, 646+ languages, GGUF F16/Q8, Stream.[1]
- `language_map.inc` содержит ровно 646 entries и `{"ru", "Russian"}`.
- Packages из `audio-cpp/audio.cpp-gguf`:
  - Q8_0 — 1,350,288,416 bytes;
  - F16/BF16 — около 1,639,548,xxx bytes.[2][14]
- Output: final merged WAV; streaming mode также выдаёт chunk WAV/PCM events.
- **Streaming означает chunked pseudo-streaming**, а не model-native low-latency token/audio generation: runtime последовательно синтезирует текстовые chunks, выдаёт по одному audio event и затем merged final WAV.[3][5]
- Важная spec/runtime коллизия: model spec помечает `reference_text` optional, но `prompt_builder.cpp` **runtime-fail** при reference audio без reference text. Следовательно, для audio.cpp clone transcript обязателен.[4]
- `mem_saver` освобождает staged graphs после фаз, но может вызвать rebuild; это не полноценный unload и не число VRAM.

---

## 4. Compatibility matrix

| Capability VOP | Qwen ASR/audio.cpp | Nemotron/audio.cpp | Qwen TTS/audio.cpp | OmniVoice/audio.cpp |
|---|---|---|---|---|
| Batch request | Совместим | Совместим | Совместим | Совместим |
| Russian | Да | Да | Да | Да |
| Context prompt | Есть, проверить semantics | Нет: locale ≠ context | instruction/speaker | design attributes, не arbitrary prose |
| Timestamps | Только с forced aligner | Words заявлены; доказать output | N/A | N/A |
| Current VOP output | Typed text receipt | Typed text receipt | WAV можно завернуть в `SynthesisResult` | WAV/PCM можно завернуть |
| Preset voices | N/A | N/A | Возможна parity, но Aiden/Sohee проверить | Нет эквивалента Qwen preset names |
| Clone без ref text | N/A | N/A | Возможно для Qwen Base, требует spike | **Регрессия:** audio.cpp OmniVoice требует transcript |
| Voice design | N/A | N/A | Новая Qwen VoiceDesign family | Есть, но Russian stability unknown |
| Native streaming | Loader support | Loader support | Нет в текущем Qwen audio.cpp session | Нет; только pseudo-streaming |
| VOP resumability/artifacts | Остаются у VOP | Остаются у VOP | Остаются у VOP | Остаются у VOP |
| Cancellation | Требует process owner | То же | То же | То же |

Главные возможные regressions:

1. OmniVoice потеряет official Python auto-transcription reference audio.
2. Qwen TTS preset voice/instruction parity не доказана.
3. ASR generic flags не гарантируют words/confidence конкретной family.
4. audio.cpp server не может прервать зависший CUDA call из userspace.
5. Different-family concurrent requests могут конкурировать за одну GPU, поскольку upstream locking — per model, а не global GPU.

---

## 5. Target architecture

```text
FasterWhisperProvider ────────────────┐
Cloud providers ──────────────────────┤
                                     ├→ existing VOP CLI/artifacts/receipts
AudioCppASRProvider ─┐                │
AudioCppTTSProvider ─┴→ LocalAudioRuntime
                         ├─ binary/build identity
                         ├─ global single-GPU lease
                         ├─ model/session load-unload
                         ├─ cancellation/process supervision
                         ├─ private inputs/temp outputs
                         └─ runtime/model/resource receipt
```

### Binary/process choice

- **Один custom binary:** предпочтителен; единый build SHA/toolchain и family registry. Но «один binary» не означает одновременно держать все weights в GPU.
- **Per-request subprocess:** принять только для first standalone spikes и failure mapping. Для chunked TTS production — отклонить: repeated model load ухудшит latency и делает 5-run stability нерепрезентативной.
- **Long-lived `audiocpp_server`:** наиболее практичный production candidate. Он сохраняет model/session, поддерживает lazy load и explicit unload endpoints.[6]
- **Но VOP нужен внешний runtime owner:** upstream server сохраняет все использованные модели до unload, сериализует только внутри одного model ID и не обеспечивает global single-GPU arbitration. `LocalAudioRuntime` должен выдавать lease, выгружать предыдущую family, контролировать health/restart и фиксировать receipts.
- **Cancellation:** soft cancel до inference; client disconnect для stream; timeout/busy guard; для зависшего CUDA — supervised process termination/restart. Нельзя обещать in-process cancellation.
- **Privacy:** loopback-only, `log_request_body=false`, no CORS, private temp directory, не писать transcript/reference/design prompt в общие logs, не auto-download weights.

---

## 6. Migration sequence и rollback

1. **Qwen3-ASR first — ACCEPT SPIKE.** Есть лучший текущий comparator: 50 cases × prompt off/on, ниже observed VRAM среди двух Python ASR и явный context contract.
2. **Nemotron ASR — ACCEPT SPIKE после Qwen, serially.** Не запускать параллельно на одной GPU.
3. **Qwen TTS CustomVoice — ACCEPT SPIKE:** Aiden и Sohee, те же тексты/instructions/artifacts.
4. **Qwen TTS Base clone — отдельный spike:** ref audio с transcript и без него.
5. **OmniVoice license/provenance gate**, затем official Python baseline → standalone `omnivoice.cpp` reference → audio.cpp OmniVoice на идентичных assets.
6. После первого ASR и первого TTS pass реализовать `LocalAudioRuntime`.
7. Продвигать family по одной, сохраняя Python provider как rollback.
8. После доказанного operational window отдельно решать removal старых Python extras. Не удалять их в интеграционном PR.

Rollback per family: feature flag/registry route обратно на Python provider → unload/restart runtime → сохранить receipts → generic offline suite. Никакого изменения Faster-Whisper/cloud.

---

## 7. Exact benchmark gates

### ASR

- Тот же `tests/fixtures/wvm_slice5_benchmark/manifest.json`, все **50 cases**.
- Сначала добавить отсутствующий same-corpus Faster-Whisper runtime baseline.
- Qwen: prompt off/on; Nemotron: no fake prompt.
- **5 полных runs** на одинаковых hashes: минимум один cold process и четыре warm/new-session repeats.
- WER/CER:
  - все 50 execution cases;
  - 18 `non_empty`;
  - language, degraded, chunked-tail, prompt-quality, no-speech/rejected subsets.
- Отдельно: hallucination/no-speech suppression, punctuation, Russian names/acronyms/numerals.
- Resource/operational: cold load, warm wall, RTF, peak VRAM/RAM, OOM, process residue, unload и return-to-baseline.
- Promotion: качество не выходит за predeclared baseline repeatability; 50/50 без crash/parser failure; receipts и hashes полны; есть материальная resource/support польза.

### TTS

Зафиксировать:

- existing Sohee report asset: `out/qwen-eval-02-sohee-report-20260801/`;
- Aiden workflow из `docs/skills/voiceover-pipeline/docs/08-workflows.md`;
- accepted public-digest script/audio/reference set — **его canonical path в repository не найден, поэтому это prerequisite DAG item**;
- один Russian 3–10 s clone reference с дословным transcript;
- OmniVoice design prompts для male/female, age, pitch, accent/style;
- long text: 2,000-char chunk boundary и полный multi-chunk digest.

Для каждого mode: **5 runs с фиксированным seed/options**.

Метрики:

- owner A/B preference и intelligibility;
- speaker similarity для clone;
- ASR-backtranscription CER/WER;
- punctuation, stress, proper names, numerals, Latin/Cyrillic mixing;
- clicks, clipping, dropped/repeated words, excess silence, chunk seams;
- duration/speed adherence;
- cold/warm latency, RTF, pseudo-stream time-to-first-audio;
- peak VRAM/RAM, cleanup/unload;
- deterministic receipt; byte identity желательна, но GPU numeric drift оценивается по заранее заданному acoustic tolerance.

**Performance сейчас не оценивалась и не заявляется.**

---

## 8. Licensing/distribution boundaries

| Объект | Boundary |
|---|---|
| audio.cpp source/binary | Apache-2.0.[7] |
| Qwen3-ASR weights | Apache-2.0.[15] |
| Qwen3-TTS Base/CustomVoice weights | Apache-2.0.[16][17] |
| Nemotron weights | OpenMDW-1.1; требует отдельной проверки bundling/distribution.[18][19] |
| OmniVoice code | Apache-2.0 |
| **Official OmniVoice weights** | **CC-BY-NC**, согласно official model card, из-за training-data constraints.[11] |
| `omnivoice.cpp` source | MIT.[12] |
| `Serveurperso/OmniVoice-GGUF` | Card декларирует Apache-2.0, но это конфликтует с official CC-BY-NC weights.[13] |
| audio.cpp OmniVoice GGUF | Repo metadata — `license:other`; provenance должен вести к official weights.[14] |

Следствие: локальный spike допустим как исследовательская операция, но **bundling, redistribution, commercial release и automatic model download OmniVoice блокируются до юридического/provenance review**. Нельзя считать производный GGUF Apache только потому, что так написано в стороннем card.

---

## 9. Classification и Kanban DAG

| Slice | Решение |
|---|---|
| Hybrid consolidation objective | **ACCEPT SPIKE** |
| Qwen ASR | **PROMOTE IF PASS** |
| Nemotron ASR | **PROMOTE IF PASS** |
| Qwen CustomVoice/Base | **PROMOTE IF PASS**, раздельно |
| OmniVoice audio.cpp | **ACCEPT SPIKE; PROMOTE IF PASS + license approval** |
| Standalone `omnivoice.cpp` production | **DEFER**; использовать как comparator |
| Per-request subprocess production | **REJECT** для chunked TTS |
| Long-lived runtime owner | **PROMOTE IF PASS** после первого ASR/TTS evidence |
| Streaming product route | **DEFER** |
| Remove Faster-Whisper/cloud | **REJECT** |
| Big-bang replacement | **REJECT** |

Предлагаемый DAG, без создания карточек:

```text
A0 Report addendum + exact source/provenance ledger
 ├─ A1 OmniVoice licensing/provenance decision
 ├─ B1 Pinned audio.cpp build + loader/package smoke
 │   ├─ B2 Qwen ASR 50×5 spike
 │   └─ B3 Nemotron ASR 50×5 spike  [serial GPU after B2]
 └─ C1 Freeze TTS assets/public digest
     ├─ C2 Qwen CustomVoice Aiden/Sohee
     ├─ C3 Qwen Base clone
     └─ C4 OmniVoice Python → standalone C++ → audio.cpp parity [after A1]

(B2 pass OR C2 pass) → D1 LocalAudioRuntime contract
D1 + each family pass → E1/E2/E3/E4 independent optional-provider promotion
all promoted + operational window → F1 decide Python-runtime retirement
```

---

## 10. Report disposition

`docs/research/2026-08-15-audio-cpp-feasibility.md` следует **не переписывать задним числом**, а дополнить датированным addendum или новой revision:

1. текущий VOP commit — `0dccdd4...`, а report header фиксирует более ранний `f3cb5db...`;
2. прежний `REJECT full migration` остаётся корректным;
3. причина reject по отсутствию Whisper не относится к новой hybrid goal;
4. recommendation `ASR-only optional provider` стала слишком узкой;
5. отсутствует OmniVoice и критичный CC-BY-NC/provenance конфликт;
6. новая recommendation: family-by-family hybrid consolidation через общий runtime owner.

## Выполнено и состояние workspace

- Проведён независимый delta-review current code, benchmark evidence и upstream exact sources.
- Inference, model downloads, weight conversion, cache/model mutation и Kanban operations не выполнялись.
- Project tracked files не изменены; `git diff` пуст.
- Начальный и финальный `git status` одинаков: pre-existing untracked `.serena/` и `C:\\/`.
- Проблемы: Context7 действительно не индексирует audio.cpp; web extractor недоступен с текущим backend, поэтому primary raw GitHub/Hugging Face API были прочитаны напрямую.

## Sources

[1] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/README.md — audio.cpp README at exact commit
[2] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/model_specs_v1/omnivoice.json — audio.cpp OmniVoice model spec
[3] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/omnivoice/session.cpp — audio.cpp OmniVoice session
[4] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/src/models/omnivoice/prompt_builder.cpp — audio.cpp OmniVoice prompt builder
[5] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/docs/models/omnivoice.md — audio.cpp OmniVoice documentation
[6] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/app/server/README.md — audio.cpp server documentation
[7] https://github.com/0xShug0/audio.cpp/blob/502b5b74bd26e9b4aed267d1776ecf131cae7215/LICENSE — audio.cpp license
[8] https://github.com/k2-fsa/OmniVoice — Official k2-fsa OmniVoice repository
[9] https://github.com/k2-fsa/OmniVoice/blob/main/docs/languages.md — Official OmniVoice languages
[10] https://github.com/k2-fsa/OmniVoice/blob/main/docs/generation-parameters.md — Official OmniVoice generation parameters
[11] https://huggingface.co/k2-fsa/OmniVoice — Official OmniVoice model card
[12] https://github.com/ServeurpersoCom/omnivoice.cpp — Standalone omnivoice.cpp
[13] https://huggingface.co/Serveurperso/OmniVoice-GGUF — Standalone OmniVoice GGUF weights
[14] https://huggingface.co/audio-cpp/audio.cpp-gguf — audio.cpp GGUF packages
[15] https://huggingface.co/Qwen/Qwen3-ASR-0.6B — Qwen3-ASR-0.6B model card
[16] https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice — Qwen3-TTS CustomVoice model card
[17] https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base — Qwen3-TTS Base model card
[18] https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b — Nemotron 3.5 ASR model card
[19] https://openmdw.ai/license/1-1 — OpenMDW License 1.1
