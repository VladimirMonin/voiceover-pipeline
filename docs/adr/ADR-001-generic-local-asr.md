# ADR-001: Generic local ASR capability seam

- Status: accepted; family/timestamp staging superseded by the 2026-08-16 hybrid migration plan
- Date: 2026-08-15
- Version: 1.1
- Scope: provider-neutral ASR contract and registry; current family/runtime decisions are governed by the hybrid plan

> **Current decision:** the generic seam remains authoritative, but its former
> text-first staging is no longer a model limitation. Qwen word timestamps use
> the separate Qwen3 Forced Aligner; Nemotron exposes native timing entries that
> must be normalized and validated. Runtime integration and rollout now follow
> [the hybrid migration plan](../plans/2026-08-16-audio-cpp-hybrid-migration-plan.md).

## Context

The repository has a mature timing-oriented path, not a generic transcription
surface. `TranscriptionProvider.transcribe(...)` returns `TimingResult`, whose
segments are immediately serialized by `build_timing_manifest` and `build_srt`.
The CLI has three independent `--timing-provider` choice lists plus hard-coded
provider branches in doctor, preflight, dispatch, and listing.

That is compatible with a runtime that truly returns segments. It is not a
safe fit for text-first ASR. In particular, Qwen3-ASR text output is not itself
proof of usable time boundaries: its forced-alignment stage is distinct. Turning
text-only output into one invented segment would recreate the behavior that the
current CLI intentionally rejects for `openrouter-whisper` timing output.

The research inputs also distinguish an offline finite-audio operation from a
live microphone session. A streaming feature requires state, partial/final
events, monotonic sample handling, endpointing, and UI ownership. It must not
be presented as another synchronous file-timing provider.

## Decision

Introduce a generic ASR seam for future local providers, while leaving the
current timing contract and its defaults untouched.

1. Generic ASR is a separate contract from `TranscriptionProvider` and
   `TimingResult`.
2. Provider behavior is declared through an explicit registry and capability
   record. Unknown IDs fail explicitly; they never fall through to
   `faster-whisper`.
3. Text is a first-class ASR result. Segment and word timing are optional,
   provenance-bearing capabilities rather than fabricated fields.
4. Qwen enters, if separately approved and proven, as offline text ASR first.
   Alignment is a later, independent stage.
5. Nemotron enters, if separately approved and a local runtime proves its
   response schema, through the same registry. It may bridge to timing only
   after its returned spans pass the timing validation contract.
6. Device and precision selection are explicit requests resolved by the
   selected provider; CPU is the initial baseline and GPU is never implicit.
7. Streaming, model downloads, cloud fallback, package publication, and live
   inference are outside this decision.

## Capability contract

The future `ASRProvider` contract should accept a finite local audio input and
return an immutable normalized result. Suggested names are illustrative; the
semantic fields are the decision.

| Contract part | Required behavior |
|---|---|
| `ASRRequest` | Audio path, optional forced language, model selection, device/compute request, and typed hints. The request identifies a finite audio item only. |
| `ASRResult` | Canonical transcript, detected/effective language, duration when known, provider/model/runtime provenance, resolved execution receipt, and optional spans. |
| `ASRSegment` | Text plus optional segment start/end. Its presence alone does not imply word alignment. |
| `ASRWordSpan` | Text with start/end and optional confidence. It is emitted only by a declared native or forced-alignment capability. |
| `ASRCapabilities` | Declarative support values for batch audio, streaming, language forcing, contextual bias, phrase boosting, segment timestamps, word timestamps, forced alignment, confidence, device modes, and compute modes. |
| `ASRProviderSpec` | Stable ID, factory/deferred-import boundary, models metadata, dependency probe, capabilities, device/compute policy, and redacted diagnostic remedy. |
| `ASRExecutionReceipt` | Actual runtime version, model revision or artifact fingerprint, resolved device, resolved compute mode, and optional execution measurements. |

Capabilities describe what a provider can really produce in the selected
runtime. A missing capability must yield a structured unsupported-capability
error, not an empty field that downstream code interprets as real data.

Initial capability rules:

- `batch_audio` is required for the first ASR providers.
- `streaming` is false in the first seam. A future `ASRSession` and event model
  is a separate ADR, not an extension of `ASRRequest`.
- Qwen text ASR declares text and contextual bias only until an aligner has
  actually produced validated spans.
- A Nemotron adapter may declare timestamps only after the chosen local
  runtime's response schema is verified with fixtures and a local run.
- `word_timestamps`, `forced_alignment`, and `confidence` are independent
  flags. No flag is inferred from the provider name.

## Prompt, context, and glossary semantics

The public generic field is not a free-form universal `prompt` string. It uses
typed ASR hints with these meanings:

| Hint | Meaning | Portable guarantee | Provider mapping |
|---|---|---|---|
| `context_text` | Short background context for the upcoming audio; soft contextual bias only. It is not a chat instruction and does not require a literal match. | The adapter may pass bounded text only when `contextual_bias` is declared. | Qwen maps it to the selected runtime's `prompt` or `context` spelling. |
| `glossary` | Curated terms, aliases, and optional source/profile ID selected by application policy. A provider may transform an approved bounded selection into contextual text. | The result records a hash/profile ID, not raw glossary terms. | Qwen may fold the selected terms into context; no accuracy promise follows. |
| `phrase_hints` | Exact phrases plus semantic strength `mild`, `normal`, or `strong`. | Available only when `phrase_boosting` is declared. Numeric vendor-specific boost values are not portable API. | Nemotron maps this only after its installed runtime API is verified. |
| `initial_prompt` | A possible compatibility mapping for a future Whisper-family adapter, not a generic user promise. | It requires an explicit adapter mapping and tests. | It is never silently substituted for Qwen context or Nemotron phrase boosting. |

Hints must be bounded and privacy-aware. Normal receipts, diagnostics, and
benchmark records store profile/version identifiers and hashes by default; they
do not log audio, raw transcript, raw context, personal names, or correction
history. Selection of terms from a larger glossary is an application concern,
not model inference behavior.

## Registration and optional dependencies

`ASRProviderSpec` becomes the single registration owner for the new ASR
surface. Parser choices, provider construction, doctor, dependency preflight,
and `list asr-providers` must read the same registry data. The registry factory
uses deferred imports so base TTS, current timings, and `list` do not import an
uninstalled ASR runtime.

Candidate optional dependencies are deliberately separated:

- `asr-qwen`: text ASR runtime only. Exact distribution names and version
  constraints remain unresolved until an approved local compatibility check.
- `asr-qwen-align`: separate alignment dependency and model stage if it has
  materially separate artifacts.
- `asr-nemotron`: only if the selected local integration is a Python runtime.
  If the approved boundary is an external `nemo-speech` executable, the spec
  probes that executable and its declared model path instead of inventing a
  Python package requirement.

Model weights, cache locations, external binaries, and model paths are not
Python extras. They need a separate acquisition/provenance policy and doctor
receipt. A missing optional dependency or required local artifact must fail
closed with a provider-specific redacted remedy. It must not inspect `.env`,
call cloud APIs, download a model, or fall back to another provider.

## Device and compute policy

The generic request carries a requested device and compute mode, such as
`auto`, `cpu`, or `cuda` and a provider-supported precision. The provider
resolves those values and returns the actual values in
`ASRExecutionReceipt`.

Rules:

- CPU is the first viable baseline and a required fallback for the initial
  local experiments.
- `auto` is provider-local resolution, never a global `torch.cuda.is_available`
  inference that claims model compatibility.
- GPU selection is opt-in in the first provider slices. No default changes to
  existing `--timing-device`, `--timing-compute`, timing model, or
  `faster-whisper` behavior are permitted.
- Doctor checks the selected provider's declared runtime/model requirements;
  CUDA availability alone is not a health proof.
- Benchmark receipts include the resolved device, compute mode, runtime version,
  model revision/fingerprint, and cold or warm state.

## Timestamp and alignment boundary

`TimingResult` remains the current voiceover timing artifact input. The generic
ASR result becomes a `TimingResult` only through an explicit bridge after it
has real, validated segment or word spans.

The bridge validates at least:

- non-negative starts and ends;
- start <= end for every span;
- monotonic ordering;
- no invalid zero-duration words unless the declared runtime semantics permit
  and tests cover them;
- expected coverage and source-audio identity;
- a declared alignment origin: `native` or `forced`.

A text-only result produces a transcript artifact, not SRT or a synthetic
single timing segment. Qwen ForcedAligner receives the exact transcript and
source audio as a separate operation; it does not silently rerun transcript
inference. Only a successful validated aligner result may use the existing
`build_timing_manifest` and `build_srt` renderers.

## CLI and migration compatibility

The existing `timings`, `generate --with-timings`, `--timing-provider`, JSON
objects, exit codes, and output safeguards remain compatible during the first
ASR slices. Neither Qwen nor Nemotron is added to an existing timing choice
until a provider proves the required capability.

A future finite-audio `transcribe` command and `list asr-providers` target are
separate from `timings`. Their exact public flags and JSON schema are frozen in
the implementation slice that adds them, then documented in
`docs/agent-cli-contract.md` and covered by deterministic tests. The initial
public ASR command need only expose audio, provider, model, language, device,
and compute. Context/glossary file UX waits for a reviewed redaction and input
schema decision; it does not start as a generic `--prompt` switch.

New generic errors should reuse the documented semantic exit-code categories
only when their meaning fits. Any new code or changed human text requires an
explicit contract update and regression tests; no undocumented broadening of
code 40 is assumed here.

## Corpus manifest and benchmark record

The corpus manifest is versioned independently of individual model runs. The
minimum `asr-corpus/v1` case schema is:

```json
{
  "schema_version": 1,
  "corpus_id": "local-asr-regression-v1",
  "case_id": "ru-noise-001",
  "tier": "P0",
  "audio": {"path": "audio/ru-noise-001.ogg", "sha256": "...", "format": "ogg/opus", "sample_rate_hz": 48000},
  "reference": {"path": "references/ru-noise-001.txt", "sha256": "...", "language": "ru", "state": "no_speech"},
  "category": "noise",
  "required_anchors": [],
  "forbidden_anchors": [],
  "license_status": "approved-or-not-verified",
  "source_provenance": "..."
}
```

The WVM Slice 5 pack is reference-only at this point. Its handoff records 72
paired Ogg/Opus plus UTF-8 reference cases and original SHA-256s, but no
redistribution grant. No asset may be copied into this repository until the WVM
owner approves provenance and redistribution. An approved subset must retain
pairing, case ID/tier/category/state/language, anchors, and original hashes.

The benchmark record is separate from the corpus case and includes fixed
configuration and measured values:

```json
{
  "benchmark_version": 1,
  "corpus_id": "local-asr-regression-v1",
  "case_id": "ru-noise-001",
  "audio_sha256": "...",
  "reference_version": "...",
  "provider_id": "...",
  "model_id": "...",
  "model_revision": "...",
  "runtime": "...",
  "runtime_version": "...",
  "device": "...",
  "compute_type": "...",
  "context_profile_hash": "...",
  "phrase_hint_profile_hash": "...",
  "timestamp_mode": "none|native|forced",
  "run_state": "cold|warm",
  "wall_s": null,
  "rtf": null,
  "peak_ram_mb": null,
  "peak_vram_mb": null,
  "wer": null,
  "cer": null,
  "term_recall": null,
  "false_insertions": null,
  "alignment_coverage": null,
  "timestamp_monotonic": null,
  "status": "not_run"
}
```

Model, pipeline, and future product/streaming benchmarks remain separate. Each
comparison pins segmentation, language, prompt/context policy, VAD, alignment,
cold/warm state, runtime, and model revision. No quality, latency, or resource
threshold is invented before a local baseline exists.

## Reconciliation ledger

| Classification | Existing baseline / research proposal | Decision and proof gate |
|---|---|---|
| ACCEPT | `TimingResult -> timings JSON + SRT` consumes required segments and current users rely on it. | Preserve it unchanged; regression tests must prove compatibility in every ASR CLI slice. |
| MODIFY | Timing registration is repeated across parser, doctor, preflight, dispatch, and list branches. | New ASR registration is registry-derived. Prove an unknown ID cannot route to FasterWhisper. |
| MODIFY | Qwen is text-first and its alignment stage is separate. | Add it behind generic text ASR; an ASR-to-timing bridge needs validated real spans. |
| DEFER | Nemotron runtime API, model artifact, phrase-boost mechanics, and offsets are not locally verified. | Choose the actual local boundary first, then prove it with fixtures and an approved offline experiment. |
| DEFER | Streaming requires session/event semantics beyond finite-audio CLI execution. | Design a separate `ASRSession`/event ADR after the batch seam and runtime evidence exist. |
| REJECT | A synthetic full-audio timing segment or a broad universal `prompt` field. | Both misstate capability; use typed hints and explicit unavailable-capability errors instead. |

## Rejected alternatives

- Add Qwen as another `--timing-provider`: rejected because text output alone
  does not satisfy the current timing/SRT contract.
- Return one synthetic full-audio segment for text ASR: rejected because it
  creates misleading timing artifacts.
- Keep registrations as parallel parser/doctor/preflight/dispatch edits:
  rejected because the current repeated branches can drift or route an unknown
  provider to FasterWhisper.
- Treat streaming as batch ASR with a flag: rejected because it hides a
  different lifecycle and event contract.
- Promise a portable numeric phrase boost or generic chat prompt: rejected
  because Qwen context, Whisper initial prompt, and Nemotron phrase boosting
  have different semantics and unverified runtime details.

## Evidence and limits

Static evidence was independently checked on 2026-08-15:

- Codebase project `home-v-code-voiceover-pipeline`, moderate index: 1,093
  nodes / 2,794 edges. `search_code("TranscriptionProvider")` found the base
  and four implementations. `trace_path(run_timings)` and
  `trace_path(_extract_timings)` map `main -> run_timings -> _extract_timings`
  and the additional `_generate_step` caller. `search_code("timing-whisper")`
  maps the current optional extra and its test/CLI consumers.
- Serena inspected `TranscriptionProvider`, `TimingResult`, `build_parser`,
  `run_timings`, `_extract_timings`, `_preflight_timing_dependency`,
  `doctor_cmd`, `list_cmd`, `build_timing_manifest`, `build_srt`, and the
  relevant CLI tests. References show `_extract_timings` is called by
  `_generate_step` and `run_timings`; all four current concrete providers
  reference `TranscriptionProvider`.
- ast-grep found four `class $NAME(TranscriptionProvider)` implementations,
  three `$PARSER.add_argument("--timing-provider", $$$)` registrations, and
  four repeated `if timing_provider == $ID` branch groups in `cli.py`.

Inputs consumed: R1 architecture audit, R2 Qwen research, R3 Nemotron
research, and R4 WVM asset inventory. The exact research attachments remain
in the Kanban board handoffs for this ADR.

This is static design evidence, not runtime proof. No Qwen/Nemotron runtime,
model download, local inference, GPU execution, benchmark, external provider,
or WVM asset license was verified by this decision.