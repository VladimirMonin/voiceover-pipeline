# Audio.cpp hybrid local runtime migration plan

> **For Hermes:** execute this plan only through the Voiceover Pipeline Kanban DAG and the `subagent-driven-development` workflow. Terra writes one slice at a time; Default independently verifies specification, code quality, runtime evidence, and semantic acceptance before the next dependent slice. Do not run parallel GPU model processes.

**Goal:** migrate Qwen3-ASR, Nemotron 3.5 ASR, Qwen3-TTS, and OmniVoice to a managed local-runtime layer using `audio.cpp` first and permitting MLX later, while keeping Faster-Whisper and all cloud providers as independent first-class routes.

**Architecture:** build the final runtime-neutral boundary first: typed VOP contracts → family providers → `LocalAudioRuntime` → a pinned `audio.cpp` driver now and an MLX driver later. Qwen uses free contextual prompting plus Qwen3-ForcedAligner timestamps; Nemotron uses its typed model prompt conditioning plus native RNN-T timestamps, with an explicit extension spike for term/hotword context. OmniVoice is mandatory scope of the first consolidated runtime release, not a future afterthought. Migration is promoted family-by-family behind explicit runtime selection and rollback; it is not a destructive big-bang cutover.

**Tech stack:** Python 3.11+, `uv`, dataclasses/ABCs, argparse, pytest, runtime-driver registry, subprocess/JSON transport, pinned `audio.cpp` C++/CMake with CUDA/CPU/Metal/Vulkan/HIP capability metadata, future MLX driver, GGUF/model artifacts, ffmpeg/ffprobe, existing VOP benchmark and artifact layers.

---

- Status: planned; implementation has not started
- Version: 1.0
- Date: 2026-08-16
- Voiceover Pipeline baseline: `ce689929096e2cd15b523303182814c5cb93254d`
- Pinned spike candidate: `audio.cpp@502b5b74bd26e9b4aed267d1776ecf131cae7215`
- Research: [audio.cpp feasibility](../research/2026-08-15-audio-cpp-feasibility.md)
- Research addendum: [hybrid consolidation](../research/2026-08-16-audio-cpp-hybrid-consolidation.md)
- Existing ASR decision: [ADR-001](../adr/ADR-001-generic-local-asr.md)

## Executive correction

The existing VOP Python adapters are text-only; the model families are not.
The previous statement that Qwen3-ASR and Nemotron “have no timestamps” mixed
adapter limitations with model/runtime capabilities.

Verified capability model:

| Family | Timestamp mechanism | Target VOP origin | Important limit |
|---|---|---|---|
| Qwen3-ASR | Separate `Qwen3-ForcedAligner-0.6B`; `return_time_stamps=True` | `forced` | Official Python streaming does not return timestamps; offline alignment is a separate pass. Current wrapper limit is 180 s per alignment chunk. |
| Nemotron 3.5 ASR | Native RNN-T emission timeline; `audio.cpp` writes decoder timestamps into `TaskResult.word_timestamps` in offline and streaming sessions | `native` | Exact `audio.cpp` code initially creates entries from tokenizer chunks. Russian subword-to-word normalization and boundary quality require a real parity test. |
| Faster-Whisper | Native segment timestamps plus word alignment | `native` | Remains a separate provider and timing fallback; it is not moved into `audio.cpp`. |

Primary evidence:

- Qwen official Context7 library: `/qwenlm/qwen3-asr`, sourced from
  <https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md>. It documents
  `forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B"` and
  `return_time_stamps=True`.
- `audio.cpp@502b5b7`:
  - `src/models/qwen3_asr/session.cpp:139-152,195-245,361-366`;
  - `src/models/qwen3_forced_aligner/processor.cpp:285-369`;
  - `assets/pipeline/qwen3_asr_subtitles.json:20-45`;
  - `src/models/nemotron_asr/decoder.cpp:53-108,456-457,559-560`;
  - `src/models/nemotron_asr/session.cpp:298-302,467-470`;
  - `include/engine/framework/runtime/session.h:104-107`.
- Owner notes:
  - `00_Inbox/4. qwen3_asr_0_6b_article_4_long_lectures_timestamps.md`;
  - `00_Inbox/6. qwen3_asr_0_6b_article_6_software_architecture.md`;
  - `00_Inbox/5. nemotron_3_5_asr_05_long_form_lectures_timestamps_diarization.md`;
  - `00_Inbox/6. nemotron_3_5_asr_06_vs_whisper_russian_benchmark_architecture.md`.

## Fixed owner decisions

1. Faster-Whisper remains installed, registered, documented, benchmarked, and
   independently selectable.
2. Cloud ASR/TTS providers remain independent and are not routed through
   `audio.cpp`.
3. The local non-Whisper target is one shared runtime boundary for:
   Qwen3-ASR, Qwen3-ForcedAligner, Nemotron ASR, Qwen3-TTS, and OmniVoice.
4. Qwen and Nemotron must expose real timestamps when the selected mode supports
   them; absence in incumbent Python adapters is not accepted as the final
   product contract.
5. Architecture is designed for all families now. Promotion remains serial and
   evidence-gated so one failing family cannot block rollback of the others.
6. Existing Python adapters remain available until their corresponding
   `audio.cpp` slice has passed quality, timing, resource, and stability gates.
7. No provider may invent timestamps, confidence, phrase boosting, streaming,
   or language semantics.
8. One GPU-heavy local model process at a time; WVM has priority.
9. Both ASR families expose model-specific prompt conditioning and timestamps,
   but their prompt semantics remain typed and distinct:
   - Qwen: free contextual text for contextual bias;
   - Nemotron: language/task prompt selected through the model prompt dictionary;
   - Nemotron term/hotword context: required migration spike/extension, because
     `audio.cpp@502b5b7` does not yet expose arbitrary phrase boosting.
10. `audio.cpp` is the first runtime driver, not the permanent provider
    abstraction. The provider and artifact contracts must permit a future MLX
    driver without renaming provider IDs or changing stored results.
11. OmniVoice is part of the first complete local-runtime release. Its live
    promotion remains evidence- and license-gated, but its contracts, inventory,
    lifecycle, CLI surface, and tests are implemented from the beginning.

## Target architecture

```text
VOP CLI / jobs / artifacts
│
├── ASRProviderRegistry
│   ├── FasterWhisperASRProvider          # remains separate
│   ├── cloud ASR providers               # remain separate
│   ├── QwenLocalASRProvider
│   │   ├── Qwen3-ASR
│   │   └── Qwen3-ForcedAligner
│   └── NemotronLocalASRProvider
│       └── native RNN-T timestamps
│
├── TTS provider registry
│   ├── cloud TTS providers               # remain separate
│   ├── QwenLocalTTSProvider
│   └── OmniVoiceLocalTTSProvider
│
└── LocalAudioRuntime
    ├── AudioCppRuntimeDriver              # first production target
    │   ├── pinned binary/build receipt
    │   ├── subprocess transport for spikes
    │   └── supervised server transport after protocol proof
    ├── PythonIncumbentDriver              # rollback window
    └── MlxRuntimeDriver                    # future, same typed contract

Shared runtime manager responsibilities:
    single-GPU lease and queue; model load/unload lifecycle;
    cancellation/timeouts; private temporary workspace;
    execution/model/build receipts; driver capability selection.
```

`AudioCppASRProvider` and `AudioCppTTSProvider` are explicitly rejected as the
public product abstraction: they would encode the current runtime into provider
IDs and make MLX a second migration. Public providers are model-family providers
over `LocalAudioRuntime`; `AudioCppRuntimeDriver` owns the current native process.

## Canonical contracts

### ASR request

Extend `ASRRequest` without adding a generic free-form prompt:

```python
ASRTimestampMode = Literal["none", "word"]

@dataclass(frozen=True)
class ASRRequest:
    audio_path: Path
    model_id: str | None = None
    language: str | None = None
    device: str = "auto"
    compute: str = "auto"
    hints: ASRHints = ASRHints()
    timestamp_mode: ASRTimestampMode = "none"
```

Rules:

- Qwen maps `context_text` to Qwen context and uses forced alignment only when
  `timestamp_mode="word"`.
- Nemotron maps a typed language/task prompt to the exact model
  `prompt_dictionary`; this is real prompt conditioning, but not Qwen-style free
  context.
- Nemotron `phrase_hints`/hotwords are a required `audio.cpp` extension spike.
  They remain capability-unavailable until the wire request, decoder behavior,
  and live term benchmark pass; the language prompt must never be mislabeled as
  phrase boosting.
- Faster-Whisper maps `timestamp_mode="word"` to its existing
  `word_timestamps=True` behavior.
- Streaming remains a separate session/event contract, not a boolean added to
  finite-file `ASRRequest`.

### ASR result

Keep `ASRResult.words: tuple[ASRWordSpan, ...]` and
`alignment_origin: Literal["native", "forced"]`. Add no second timestamp model.
Every adapter must normalize its wire output into this contract.

Required validation:

- start/end are finite, non-negative, monotonic, and inside source duration;
- speech output with requested timestamps cannot silently return an empty word
  list;
- no-speech output cannot fabricate words;
- normalized concatenated words must correspond to the returned transcript
  under a documented punctuation/whitespace normalization;
- Qwen returns `alignment_origin="forced"`;
- Nemotron and Faster-Whisper return `alignment_origin="native"`;
- confidence remains `None` when the runtime only emits placeholder `0.0`.

### Runtime transport

```python
class LocalAudioRuntimeDriver(Protocol):
    def invoke(self, request: LocalRuntimeRequest) -> LocalRuntimeResponse: ...
    def cancel(self, request_id: str) -> None: ...
    def close(self) -> None: ...

class LocalAudioRuntime:
    def execute_asr(self, request: LocalASRRequest) -> LocalASRResponse: ...
    def execute_tts(self, request: LocalTTSRequest) -> LocalTTSResponse: ...
    def unload(self, family: str | None = None) -> None: ...
```

`AudioCppRuntimeDriver` implements this protocol first. Its subprocess transport
proves the binary and schemas; the managed server transport may replace it
internally after one ASR and one TTS family pass. A future `MlxRuntimeDriver`
implements the same protocol, so providers and artifacts do not change.

## Migration and rollback policy

Every migrated provider accepts an explicit runtime choice during the overlap
window:

```text
--local-runtime python
--local-runtime audio-cpp
--local-runtime auto
```

Initial default remains `python`. `auto` may select `audio-cpp` only after the
specific family is promoted. A provider-level rollback changes selection, not
contracts or stored artifacts. Python code and dependency extras are removed
only by a later explicit owner decision after an operational observation window.

Receipts record:

- provider ID and family;
- VOP version;
- `audio.cpp` commit and build hash;
- transport kind;
- model ID, file hashes, quantization, and license/provenance;
- resolved device/backend;
- request timestamp mode and alignment origin;
- wall time, RTF, peak RAM/VRAM where available;
- exit/cancellation state;
- redacted private temporary paths.

## Implementation DAG

```text
P0 documentation correction / ADR reconciliation
  └─ P1 canonical timestamp + runtime contracts
      ├─ P2 pinned build and wire-schema spike
      │   ├─ P3 Qwen ASR + Forced Aligner parity
      │   ├─ P4 Nemotron native timestamp parity
      │   ├─ P6 Qwen TTS parity
      │   └─ P7 OmniVoice first-release integration + live gate
      └─ P5 generic timing artifact bridge
          ├─ P3
          └─ P4

P3 + P4 + P5 ──> P8 ASR rollout/rollback integration
P6 + P7 ───────> P9 TTS rollout/rollback integration
P3 + P6 ───────> P10 supervised runtime/server promotion
P8 + P9 + P10 ─> P11 full regression, semantic review, release gate
```

Only P3/P4 research runners may be prepared independently. GPU live runs are
serial. Product writers are serial in the shared worktree.

## Tasks

### Task 1: Reconcile governing documentation

**Objective:** replace the incorrect model-level “no timestamps” implication
with the verified adapter/runtime distinction before code changes.

**Files:**

- Modify: `docs/research/2026-08-16-audio-cpp-hybrid-consolidation.md`
- Modify: `docs/adr/ADR-001-generic-local-asr.md`
- Modify: `docs/plans/2026-08-15-generic-local-asr-implementation-plan.md`
- Modify: `docs/README.md`
- Test: documentation links and citation/source checks

**Steps:**

1. Add a failing documentation assertion that both target ASR families have a
   verified timestamp route and that incumbent adapters remain text-only.
2. Run the repository documentation/link checker and confirm failure on stale
   wording.
3. Mark ADR-001’s text-only statements as historical incumbent facts; add this
   plan as the superseding migration decision.
4. Record Qwen forced alignment and Nemotron native timestamp evidence with
   exact source paths listed above.
5. Run link checks and `git diff --check`; expected: PASS.
6. Commit only the documentation scope.

### Task 2: Add timestamp intent and runtime selection to typed contracts

**Objective:** let callers request real word timestamps without coupling the
request to Whisper or `audio.cpp`.

**Files:**

- Modify: `src/voiceover_pipeline/models.py`
- Modify: `src/voiceover_pipeline/providers/base.py`
- Test: `tests/test_asr_contract.py`

**Steps:**

1. Write failing tests for `timestamp_mode="none"|"word"`, invalid values,
   requested-but-empty timestamp results, in-bounds spans, and origin rules.
2. Run `uv run --offline pytest -q tests/test_asr_contract.py`; expected: FAIL.
3. Add the smallest typed request/result validation needed by the tests.
4. Preserve existing constructor compatibility with default `"none"`.
5. Rerun the focused suite; expected: PASS.
6. Commit contract and tests.

### Task 3: Create runtime-neutral `LocalAudioRuntime` and the `audio.cpp` driver

**Objective:** establish one model-family runtime contract with `audio.cpp` as
the first driver and an explicit future MLX extension point before family
adapters are implemented.

**Files:**

- Create: `src/voiceover_pipeline/local_runtime/__init__.py`
- Create: `src/voiceover_pipeline/local_runtime/contracts.py`
- Create: `src/voiceover_pipeline/local_runtime/registry.py`
- Create: `src/voiceover_pipeline/local_runtime/manager.py`
- Create: `src/voiceover_pipeline/local_runtime/drivers/audio_cpp.py`
- Create: `src/voiceover_pipeline/local_runtime/transports/subprocess.py`
- Test: `tests/test_audio_cpp_contracts.py`
- Test: `tests/test_audio_cpp_runtime.py`

**Steps:**

1. Write fixtures for successful ASR/TTS responses, malformed output, non-zero
   exit, timeout, cancellation, missing binary, and redacted diagnostics.
2. Run focused tests; expected: FAIL because the package does not exist.
3. Implement model-family typed request/response models, a runtime-driver
   protocol, registry, and strict `audio.cpp` codec parsing.
4. Add a fake second driver in tests to prove that provider-facing contracts do
   not contain `audio.cpp` CLI fields and can host MLX later.
5. Implement subprocess invocation with bounded timeout, no shell, private temp
   directory, captured stderr, and process-group cancellation.
6. Do not parse generic human CLI text when a stable file/JSON output exists.
7. Add runtime receipts and deterministic dependency health without inference.
8. Rerun focused tests; expected: PASS.
9. Commit runtime seam only.

### Task 4: Add pinned build and model inventory

**Objective:** make the native runtime reproducible and inspectable without
implicitly downloading models.

**Files:**

- Create: `scripts/build_audio_cpp.py`
- Create: `src/voiceover_pipeline/audio_cpp/inventory.py`
- Create: `docs/audio-cpp-runtime.md`
- Modify: `pyproject.toml` only if a build helper dependency is required
- Test: `tests/test_audio_cpp_inventory.py`
- Test: `tests/test_audio_cpp_build_receipt.py`

**Steps:**

1. Write failing tests for exact commit, compiler/CUDA flags, binary hash, model
   family, model hash, quantization, license, and provenance.
2. Implement a build plan/receipt generator pinned to `502b5b74…`; build output
   lives outside Git.
3. Add `doctor`-safe inventory probes that never load weights.
4. Inventory backend support explicitly: `audio.cpp` currently provides CPU,
   CUDA, HIP/ROCm, Vulkan, and Metal; it does not claim MLX.
5. Record the future MLX driver as `not installed/not implemented`, not as an
   `audio.cpp` backend.
6. Validate CPU and CUDA build commands in an approved build card.
7. Record actual binary dependencies and CUDA architecture for RTX 3060.
8. Reject a dirty/unpinned source tree and mismatched binary hash.
9. Run tests and a binary `--help`/model-list smoke; expected: PASS.
10. Commit scripts, tests, and docs; do not commit binaries or weights.

### Task 5: Implement Qwen ASR with mandatory optional aligner path

**Objective:** provide Qwen text and real word timestamps through one provider.

**Files:**

- Create: `src/voiceover_pipeline/providers/audio_cpp_qwen_asr.py`
- Modify: `src/voiceover_pipeline/providers/asr_registry.py`
- Modify: `src/voiceover_pipeline/cli.py`
- Test: `tests/test_audio_cpp_qwen_asr.py`
- Extend: `tests/test_asr_cli.py`

**Steps:**

1. Write failing fixture tests for text-only, context, forced language, aligned
   words, missing aligner, malformed words, chunk offsets, and empty speech
   alignment.
2. Assert `timestamp_mode="word"` maps to
   `qwen3_asr.forced_aligner_model_path` and requests timestamp output.
3. Assert absence of the aligner fails with a precise remediation, not a silent
   text-only downgrade.
4. Map aligned entries to `ASRWordSpan` and `alignment_origin="forced"`.
5. Keep streaming out of this finite-file provider.
6. Run focused tests; expected: PASS.
7. Commit fixture-level integration before live inference.

### Task 6: Prove Qwen timestamp parity on real assets

**Objective:** compare `audio.cpp` Qwen+aligner against the incumbent Python
Qwen+official aligner route on identical bytes.

**Files:**

- Modify: `src/voiceover_pipeline/asr_benchmark.py`
- Modify: `tests/test_asr_benchmark.py`
- Create runtime artifact under ignored `out/asr-benchmarks/`

**Steps:**

1. Extend the benchmark schema with timestamp mode, alignment origin, word count,
   coverage, zero-duration count, monotonicity, boundary MAE/p95, and drift.
2. Use the existing 50 WVM cases for transcript/resource comparison.
3. Add a manually aligned Russian timing subset; do not use another ASR output
   as reference truth.
4. Run Python Qwen text+official aligner and `audio.cpp` Qwen+aligner in controlled
   cold/warm order.
5. Run five consecutive `audio.cpp` repetitions without crash/leak.
6. Verify long-input chunk offsets and the 180-second safe alignment boundary.
7. Promotion gate:
   - 50/50 requests complete;
   - all speech cases requesting timestamps return non-empty in-bounds monotonic
     words;
   - no-speech cases do not fabricate timed words;
   - no material WER/CER/term regression from incumbent Qwen;
   - timing boundary quality is measured and acceptable for SRT/navigation;
   - no timestamp/parser/chunk-offset failures;
   - unload returns GPU memory near the measured baseline.
8. Record PASS/FAIL; do not promote on partial evidence.

### Task 7: Implement Nemotron native timestamp normalization

**Objective:** expose Nemotron native RNN-T timestamps as real VOP word spans.

**Files:**

- Create: `src/voiceover_pipeline/providers/audio_cpp_nemotron_asr.py`
- Create: `src/voiceover_pipeline/audio_cpp/nemotron_words.py`
- Modify: `src/voiceover_pipeline/providers/asr_registry.py`
- Test: `tests/test_audio_cpp_nemotron_asr.py`
- Test: `tests/test_nemotron_word_normalization.py`

**Steps:**

1. Write failing fixtures containing SentencePiece/metaspace chunks, punctuation,
   Cyrillic, Latin technical terms, multiple chunks on one frame, zero-duration
   spans, and chunk-boundary overlap.
2. Preserve raw runtime entries in receipt/debug artifacts while merging only
   contiguous tokenizer pieces that form one canonical word.
3. Do not set confidence from `0.0` placeholders.
4. Map output to `ASRWordSpan` with `alignment_origin="native"`.
5. Validate normalized text equivalence, monotonicity, source bounds, and no
   out-of-keep-span entries.
6. Map the supported language/task model prompt through the prompt dictionary
   and verify it selects the expected prompt ID.
7. Add a bounded extension spike for Nemotron term/hotword context. If the exact
   `audio.cpp` surface lacks it, implement or upstream the smallest typed request
   extension rather than pretending locale is contextual prompting.
8. Keep arbitrary phrase boosting capability disabled until fixture, decoder,
   and live term tests pass.
9. Run focused tests; expected: PASS.
10. Commit fixture-level integration before live inference.

### Task 8: Prove Nemotron timestamp parity on real assets

**Objective:** validate transcript quality, native timestamp granularity, and
resource behavior against the incumbent Nemotron baseline and Faster-Whisper
reference mode.

**Files:**

- Modify: `src/voiceover_pipeline/asr_benchmark.py`
- Modify: `tests/test_asr_benchmark.py`
- Create runtime artifact under ignored `out/asr-benchmarks/`

**Steps:**

1. Run the same 50 WVM cases and the same manually aligned timing subset.
2. Measure WER/CER/terms/no-speech separately from timestamp quality.
3. Measure word reconstruction failures, zero-duration entries, boundary
   MAE/p95, signed error, and long-record drift.
4. Run five consecutive repetitions and verify cancellation/unload.
5. Promotion gate:
   - 50/50 requests complete;
   - all requested speech cases return normalized timed words;
   - no fabricated no-speech words;
   - no material transcript regression from incumbent Nemotron;
   - native boundaries are acceptable for SRT/navigation;
   - tokenizer-chunk merging is deterministic;
   - no crash, leak, out-of-span word, or cumulative drift defect.
6. Record that Nemotron timestamps are native RNN-T alignment, not forced
   alignment and not phoneme-accurate boundaries.

### Task 9: Unify generic ASR results with timing artifacts

**Objective:** allow Qwen, Nemotron, and Faster-Whisper to generate the same
validated timing JSON/SRT without removing legacy CLI compatibility.

**Files:**

- Create: `src/voiceover_pipeline/asr_timing_bridge.py`
- Modify: `src/voiceover_pipeline/artifacts.py`
- Modify: `src/voiceover_pipeline/cli.py`
- Test: `tests/test_asr_timing_bridge.py`
- Extend: `tests/test_generation_stability.py`
- Extend: `tests/test_cli_json_contract.py`

**Steps:**

1. Write failing tests converting native and forced `ASRResult.words` to current
   timing artifacts.
2. Reject text-only results, missing origins, malformed spans, and transcript/
   word mismatches.
3. Build segment/cue groups from words using existing artifact rules.
4. Route new provider IDs through registry-derived selection.
5. Keep current `timings --timing-provider faster-whisper` behavior and output
   byte-compatible.
6. Add generic provider selection without replacing the Faster-Whisper default.
7. Run focused and full offline suites; expected: PASS.
8. Commit bridge and compatibility tests.

### Task 10: Implement shared GPU lease and lifecycle owner

**Objective:** prevent local ASR/TTS families from competing for one GPU and
make unload/cancellation observable.

**Files:**

- Create: `src/voiceover_pipeline/local_runtime/gpu_lease.py`
- Create: `src/voiceover_pipeline/local_runtime/lifecycle.py`
- Modify: `src/voiceover_pipeline/local_runtime/manager.py`
- Test: `tests/test_audio_cpp_gpu_lease.py`
- Test: `tests/test_audio_cpp_lifecycle.py`

**Steps:**

1. Write failing concurrency tests for FIFO ownership, cancellation, stale owner,
   process death, timeout, and release.
2. Implement one process-wide lease plus cross-process lock metadata without
   storing audio/text.
3. Add preflight hooks for free VRAM, utilization, temperature, and Xid policy.
4. Add model unload and worker restart paths.
5. Ensure WVM-owned GPU work is never killed; the VOP job waits or fails closed.
6. Run deterministic tests with fake probes; expected: PASS.
7. Live-test only in an approved card with a fresh GPU gate.
8. Commit lifecycle owner.

### Task 11: Migrate Qwen3-TTS through the shared runtime

**Objective:** preserve existing Sohee/Aiden workflows while replacing only the
local runtime beneath the provider.

**Files:**

- Create: `src/voiceover_pipeline/providers/audio_cpp_qwen_tts.py`
- Modify: `src/voiceover_pipeline/config.py`
- Modify: `src/voiceover_pipeline/cli.py`
- Test: `tests/test_audio_cpp_qwen_tts.py`
- Extend: existing Qwen TTS provider/generation tests

**Steps:**

1. Write fixtures for CustomVoice, Base cloning, VoiceDesign, instructions,
   sample rate, output WAV, cancellation, and malformed audio.
2. Preserve the existing `qwen-local` public provider behavior during overlap;
   select Python or `audio.cpp` internally.
3. Keep mode-specific capabilities explicit; do not flatten CustomVoice,
   cloning, and VoiceDesign into one invented prompt.
4. Run existing Sohee/Aiden/public digest fixtures against both routes.
5. Live gate: Russian pronunciation, punctuation, names/numbers, long text,
   chunk seams, RTF, time-to-first-audio, RAM/VRAM, and five-run stability.
6. Promote only if artifacts and operational behavior pass; otherwise default
   remains Python.

### Task 12: Add OmniVoice as a mandatory first-release TTS family

**Objective:** ship OmniVoice synthesis, cloning, design, and controls in the
first complete shared-runtime release without pretending it is Qwen TTS or
redistributing unclear weights.

**Files:**

- Create: `src/voiceover_pipeline/providers/audio_cpp_omnivoice_tts.py`
- Create: `docs/omnivoice-local-tts.md`
- Modify: TTS registry/config/CLI owner files identified during implementation
- Test: `tests/test_audio_cpp_omnivoice_tts.py`

**Steps:**

1. Freeze separate capabilities for auto voice, cloning, design, control,
   pseudo-streaming, long-form, and output format.
2. Register OmniVoice contracts, model inventory, doctor status, and CLI listing
   in the initial runtime foundation; do not wait for Qwen TTS promotion.
3. Require reference audio plus exact reference transcript for the `audio.cpp`
   cloning route; fail clearly when transcript is absent.
4. Preserve private reference audio in a restricted temporary workspace and
   delete it after receipt finalization.
5. Keep official Python OmniVoice, standalone `omnivoice.cpp`, and `audio.cpp`
   family names/provenance distinct in receipts.
6. Run Russian clone/design/control and long-form tests, including speaker
   similarity and back-transcription quality.
7. Treat pseudo-streaming as chunked sequential synthesis, not native streaming.
8. Block automatic download, bundling, redistribution, and commercial promotion
   until the official-weight CC-BY-NC boundary and derivative GGUF provenance
   receive explicit approval.
9. The first consolidated runtime release is incomplete until OmniVoice either
   passes or has an explicit owner-approved retained-blocker decision.

### Task 13: Add managed server transport after one ASR and one TTS PASS

**Objective:** remove repeated model cold starts without exposing a fragile daemon
as the initial integration boundary.

**Files:**

- Create: `src/voiceover_pipeline/local_runtime/transports/server.py`
- Modify: `src/voiceover_pipeline/local_runtime/drivers/audio_cpp.py`
- Modify: `src/voiceover_pipeline/local_runtime/manager.py`
- Test: `tests/test_audio_cpp_server_transport.py`
- Test: `tests/test_audio_cpp_transport_parity.py`

**Steps:**

1. Freeze the request/response schema from proven subprocess artifacts.
2. Write transport-parity tests using identical ASR/TTS fixtures.
3. Add supervised start/readiness/health/restart/shutdown behavior.
4. Add bounded request queue, cancellation, and model unload.
5. Ensure server logs never contain audio bytes, transcripts, reference audio,
   secrets, or unrestricted absolute paths.
6. Verify five family switches without GPU leakage.
7. Keep subprocess as diagnostic fallback.
8. Promote server transport only if parity and lifecycle tests pass.

### Task 14: Provider rollout and rollback

**Objective:** switch defaults family-by-family without removing working routes.

**Files:**

- Modify: `src/voiceover_pipeline/config.py`
- Modify: `src/voiceover_pipeline/cli.py`
- Modify: provider registries
- Extend: CLI/config/doctor/list tests

**Steps:**

1. Add `python|audio-cpp|auto` selection with existing Python defaults.
2. Make `doctor` report binary, build hash, model readiness, aligner readiness,
   GPU lease, and supported capabilities without inference.
3. Promote Qwen ASR, Nemotron ASR, Qwen TTS, and OmniVoice independently.
4. Verify changing one family does not affect Faster-Whisper or cloud routes.
5. Exercise explicit rollback after successful and failed requests.
6. Keep stored artifacts readable regardless of runtime choice.
7. Run full offline suite and approved live smoke matrix.
8. Commit each provider promotion separately.

### Task 15: Final semantic and release gate

**Objective:** prove the complete hybrid system matches owner intent and remains
reversible.

**Files:**

- Modify documentation only where runtime evidence now exists
- No cleanup/removal of Python adapters in this task

**Steps:**

1. Run `uv run --offline pytest -q` and report actual counts.
2. Run `git diff --check` and staged-manifest verification.
3. Run doctor/list/transcribe/timings/generate compatibility checks.
4. Run the approved ASR/TTS live matrix serially after a fresh GPU gate.
5. Independently review timestamps for both Qwen and Nemotron.
6. Verify Faster-Whisper remains selectable and unchanged.
7. Verify cloud providers import/list/doctor paths remain unchanged.
8. Verify rollback to each incumbent Python route.
9. Require exact `SEMANTIC_PASS` before commit/push.
10. Push through the existing one-shot GitHub credential workflow and verify
    local HEAD, upstream SHA, and GitHub remote SHA.

## Benchmark matrix

### ASR transcript and resource matrix

Use identical bytes and existing references:

| Route | Context | Timestamps | Repetitions |
|---|---:|---:|---:|
| Python Qwen | off/on | none/official aligner | controlled cold/warm + 5 stability |
| audio.cpp Qwen | off/on | none/forced aligner | controlled cold/warm + 5 stability |
| Python Nemotron | locale | current text baseline | controlled cold/warm |
| audio.cpp Nemotron | locale | native | controlled cold/warm + 5 stability |
| Faster-Whisper | Whisper prompt rules only | native | retained reference/fallback |

Report separately:

- 18 ordinary speech cases: WER, CER, term recall/precision;
- full 50 cases: state/no-speech/noise/pause diagnostics;
- timestamp subset: coverage, monotonicity, boundary MAE/p95, signed error,
  zero-duration rate, out-of-bounds rate, and drift;
- cold/warm wall time and RTF;
- peak RAM/VRAM and post-unload baseline;
- crash, OOM, parser, cancellation, and leak counts.

Do not use one mixed WER as universal quality. Do not attribute warm-run speed to
context or timestamps without controlled order.

### TTS matrix

Use existing Sohee/Aiden/public digest workflows plus fixed Russian test texts:

- CustomVoice parity;
- Base cloning with 3–10 s reference and exact transcript;
- VoiceDesign male/female/age/pitch/style;
- OmniVoice auto/clone/design/control;
- at least one 2000+ character long-form input;
- punctuation, numbers, names, abbreviations, Cyrillic/Latin code-switching;
- output sample rate/container, clipping, clicks, leading/trailing silence,
  chunk seams, omissions, repetitions;
- speaker similarity and blind listening;
- back-transcription WER/CER as a diagnostic, not a sole quality score;
- RTF, time-to-first-audio, RAM/VRAM, five-run stability, cancellation, unload.

## Promotion classifications

| Slice | Classification |
|---|---|
| Runtime-neutral `LocalAudioRuntime` contracts and `audio.cpp` spike driver | ACCEPT |
| Qwen3-ASR + Forced Aligner | PROMOTE IF PASS |
| Nemotron native timestamps | PROMOTE IF PASS after subword normalization proof |
| Generic timing bridge | ACCEPT after both timestamp fixtures pass |
| Qwen3-TTS | PROMOTE IF PASS |
| OmniVoice contracts/inventory/provider | ACCEPT as first-release scope |
| OmniVoice live default | PROMOTE IF PASS plus license/provenance approval |
| Future MLX runtime driver | ACCEPT extension seam now; implementation DEFER |
| Managed server transport | DEFER until one ASR and one TTS family pass |
| Production streaming sessions | DEFER to a separate session API slice |
| Faster-Whisper removal | REJECT |
| Cloud provider removal | REJECT |
| Big-bang default switch | REJECT |
| Python adapter deletion before observation window | REJECT |

## Stop criteria

Stop and retain the incumbent route for the affected family when any of these is
true:

- pinned CUDA build is not reproducible on RTX 3060;
- timestamp output is empty, malformed, out of bounds, non-monotonic, or cannot
  be normalized deterministically;
- Qwen loses contextual bias or reliable forced alignment;
- Nemotron tokenizer chunks cannot be converted to stable word spans;
- transcript/TTS quality materially regresses from the incumbent route;
- no meaningful resource, deployment, or maintenance gain remains;
- five-run stability fails, GPU memory does not return, cancellation is unsafe,
  or Xid errors recur;
- wire schema or exit behavior cannot be wrapped without leaking runtime details
  into VOP contracts;
- OmniVoice provenance/license is insufficient for the intended distribution;
- integration requires weakening Faster-Whisper/cloud behavior.

## Documentation synchronization required during execution

Update only after the corresponding tests/runtime evidence exist:

- `docs/agent-cli-contract.md` — runtime selection, timestamps, errors;
- `docs/artifacts-and-analysis.md` — canonical words/origin/receipts;
- `docs/whisper-timing.md` — Faster-Whisper remains first-class, not deprecated;
- `docs/qwen-local-tts.md` — Python/audio.cpp overlap and rollback;
- `docs/audio-cpp-runtime.md` — build, doctor, models, privacy, lifecycle;
- `docs/omnivoice-local-tts.md` — exact family/license/capabilities;
- `docs/README.md` — links to implemented docs;
- research addendum and ADR — facts vs target state.

## Definition of done

The migration program is complete only when:

1. all four local non-Whisper families use the common runtime boundary or have a
   documented retained-incumbent decision backed by failed gates;
2. Qwen and Nemotron both return validated timestamps in supported finite-file
   modes;
3. Faster-Whisper remains independently selectable and passes existing timing
   regressions;
4. cloud providers remain independently selectable;
5. every promoted family has explicit runtime selection and tested rollback;
6. GPU ownership, unload, cancellation, receipts, and privacy are verified;
7. ASR/TTS parity artifacts and five-run stability evidence exist;
8. full tests, independent review, semantic acceptance, staged manifest, commit,
   push, and remote SHA verification pass.
9. provider-facing contracts have no `audio.cpp`-specific CLI fields and a fake
   second driver proves the future MLX extension seam.
10. OmniVoice is present in contracts, inventory, doctor, CLI, tests, and the
    first-release acceptance decision rather than left as an unspecified future
    feature.
