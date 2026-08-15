# Generic local ASR implementation plan

> **For Hermes:** Execute only through the downstream Kanban implementation card, one verified slice at a time. Do not commit, download models, or run live inference unless that card explicitly authorizes it.

**Goal:** introduce a provider-neutral, offline-first ASR seam without changing existing timing behavior.

**Architecture:** keep the segment-required `TimingResult` path intact and add a separate typed ASR protocol plus registry. Text ASR stays distinct from timing/alignment; adapters use deferred imports and declare their actual capabilities.

**Tech stack:** Python 3.11+, dataclasses/ABCs, argparse, pytest, uv; all initial tests use mocks, temporary paths, and sanitized fixtures.

---

- Status: planned; implementation has not started
- Version: 1.0
- Date: 2026-08-15
- Governing decision: [ADR-001](../adr/ADR-001-generic-local-asr.md)

## Goal and boundaries

Build one generic, offline-first ASR extension seam that can host local Qwen
text ASR and a future local Nemotron adapter without weakening existing
voiceover timing behavior.

This plan does not authorize adapter implementation, package installation,
model acquisition, model download, cloud calls, GPU work, benchmark execution,
release, commit, or push. Every implementation slice stays offline and uses
fixtures/mocks until a separately approved local runtime experiment.

The fixed non-goals for the first series are microphone capture, streaming
partial events, VAD/session ownership, GUI integration, automatic model
download, cloud fallback, and copying WVM assets without an owner-approved
license/provenance decision.

## Existing owner map

The planned changes are constrained by the current code, not by a speculative
rewrite:

| Owner | Current role | Planning consequence |
|---|---|---|
| `src/voiceover_pipeline/providers/base.py:15` — `TranscriptionProvider` | Timestamp-first protocol returning `TimingResult`. | Preserve it for current timings; add a distinct generic ASR protocol. |
| `src/voiceover_pipeline/models.py:68` — `TimingResult` | Required segments plus execution provenance. | Do not expand it to fake text-only timing. |
| `src/voiceover_pipeline/cli.py:161` — `build_parser` | Repeats timing-provider choices for generate, timings, and doctor. | New ASR registration reads a single registry; old choices stay compatible. |
| `cli.py:731` — `run_timings` and `cli.py:1273` — `_extract_timings` | Finite audio -> provider -> JSON/SRT. | Do not route text-only ASR here. |
| `cli.py:863` — `doctor_cmd`; `cli.py:1165` — `_preflight_timing_dependency`; `cli.py:1067` — `list_cmd` | Separate hard-coded timing checks/list metadata. | Do not copy this pattern into the ASR surface. |
| `src/voiceover_pipeline/artifacts.py:116` / `:138` | Render `TimingResult.segments` to timing JSON and SRT. | Permit only validated real spans to use this bridge. |
| `pyproject.toml:43` | Existing `timing-whisper` and `voiceover-qwen` extras. | Keep ASR extras separate and deferred. |
| `tests/test_cli_json_contract.py` and `tests/test_generation_stability.py` | CLI JSON and optional-dependency regression seams. | Extend deterministically; no real provider/runtime tests. |

## Sequential slices

Each slice must pass its listed acceptance checks before the next begins. Do
not fold later adapter, alignment, or benchmark work into an earlier slice.

For every code-producing slice, use this fixed TDD cycle: (1) add the named
failing test, (2) run its focused `uv run --offline pytest -q <path>` command
and observe failure, (3) make the smallest implementation in the listed files,
and (4) rerun the focused command to PASS before broadening scope. No step
silently changes `uv.lock`, reads `.env`, or makes a network call.

### S0 — Contract tests and model types

Purpose: introduce the generic concepts with no provider runtime and no CLI
surface.

Planned files:

- `src/voiceover_pipeline/models.py`: add immutable generic ASR request,
  result, segment/word span, capabilities, execution receipt, and typed hints.
- `src/voiceover_pipeline/providers/base.py`: add a separate generic ASR
  protocol; leave `TranscriptionProvider` unchanged.
- `tests/test_asr_contract.py`: new offline contract tests.

Acceptance:

- A text-only result is valid and has no timestamp capability.
- Native and forced alignment origins are distinguishable.
- Span validation rejects negative, reversed, and non-monotonic values.
- Context, glossary, and phrase hints remain typed and do not create a generic
  free-form prompt field.
- Existing timing result/artifact tests stay unchanged and pass.

Focused command after implementation:

`uv run --offline pytest -q tests/test_asr_contract.py`

### S1 — Registry and dependency boundary

Purpose: make new provider registration single-source-of-truth before any
adapter exists.

Planned files:

- `src/voiceover_pipeline/providers/asr_registry.py`: `ASRProviderSpec`,
  explicit lookup, deferred factory, declared capabilities, and dependency
  health result.
- `src/voiceover_pipeline/providers/__init__.py`: export only stable generic
  registry/protocol symbols as needed.
- `tests/test_asr_registry.py`: registry-only tests.

Acceptance:

- Lookup of an unknown ASR ID raises a structured error; it never selects
  FasterWhisper or a cloud provider.
- Registry/listing code imports without Qwen/Nemotron packages, runtimes, or
  model weights.
- Each declared dependency failure has one redacted remediation string and
  does not inspect `.env`.
- A spec exposes actual supported device/compute and timestamp capabilities;
  absence is represented explicitly.

Focused command after implementation:

`uv run --offline pytest -q tests/test_asr_registry.py tests/test_asr_contract.py`

### S2 — Read-only ASR CLI plumbing

Purpose: expose a finite-audio ASR command whose behavior is safely testable
without a real adapter.

Planned files:

- `src/voiceover_pipeline/cli.py`: add a separate `transcribe` command,
  `doctor --with-asr --asr-provider`, and `list asr-providers` only after the
  parser consumes the registry.
- `tests/test_cli_json_contract.py`: JSON/stdout/stderr and exit-code cases.
- `docs/agent-cli-contract.md`: freeze exact command, flags, JSON fields, and
  semantic error categories only when tests prove them.

Acceptance:

- `--json` emits exactly one object to stdout for success and error.
- `transcribe` supports audio, provider, model, language, device, and compute;
  the initial command does not add unreviewed raw context/glossary flags.
- `timings`, `generate --with-timings`, their flags/defaults, and their output
  file behavior are byte-for-byte compatible in targeted regression tests.
- `doctor` checks only the selected ASR spec and does not claim runtime health
  merely because CUDA or torch is present.
- Listing derives IDs/capabilities from the registry rather than another
  hand-maintained provider list.

Focused commands after implementation:

`uv run --offline pytest -q tests/test_cli_json_contract.py tests/test_asr_registry.py`

`uv run --offline pytest -q tests/test_generation_stability.py`

### S3 — Qwen text-ASR adapter, fixture-first

Purpose: add the first candidate as a text-only, deferred-import local
provider, without an aligner or timing bridge.

Planned files:

- `src/voiceover_pipeline/providers/qwen_asr_local.py`: deferred import and
  normalized response adapter.
- `pyproject.toml`: a distinct `asr-qwen` extra only after approved resolver
  evidence establishes exact packages and versions.
- `tests/test_qwen_asr_provider.py`: mocked import/runtime-response fixtures.

Acceptance:

- Base CLI and all existing TTS/timing commands start with no Qwen runtime.
- A selected missing provider fails closed with a tested install remedy.
- `context_text` maps once to the selected official runtime spelling
  (`prompt` or `context`); it is contextual bias, not instruction following.
- Effective language, model/runtime identity, resolved device/compute, and
  transcript are normalized into the generic result.
- The adapter declares no timestamp/alignment capability and cannot request
  SRT output.
- No model, package download, GPU operation, or live inference is part of
  unit tests.

Focused command after implementation:

`uv run --offline pytest -q tests/test_qwen_asr_provider.py tests/test_asr_contract.py tests/test_asr_registry.py`

### S4 — Local runtime experiment gate

Purpose: decide whether an already provisioned Qwen runtime is viable on a
specific machine. This is a separate owner-approved execution card, not an
implicit continuation of S3.

Preconditions:

- A model/runtime artifact is already present, approved, and identified by
  version/revision/hash.
- CPU baseline is available; any GPU run has explicit resource approval and
  does not conflict with WVM priority.
- One fixed short local audio item, explicit language, and a short sanitized
  context profile are ready.

Record only a redacted receipt: command configuration, runtime/model identity,
resolved device/compute, wall time, RTF, peak RAM/VRAM where available, output
schema, and exit status. The experiment is a healthcheck, not a quality claim.

### S5 — Alignment bridge, only after separate proof

Purpose: connect a proven source of real timings to the existing timing
artifacts without changing text-only ASR behavior.

Candidates:

- Qwen ForcedAligner consuming the exact ASR transcript and source audio.
- Nemotron only if the chosen local runtime proves native segments/words with a
  stable schema.

Planned files after proof exists:

- dedicated provider/alignment adapter module;
- a narrow ASR-to-`TimingResult` bridge;
- `tests/test_asr_timing_bridge.py` plus timing artifact regressions;
- provider-specific optional extra or external-runtime doctor probe.

Acceptance:

- The bridge requires declared alignment origin and validates span ordering,
  non-negative duration, monotonicity, coverage, and source-audio identity.
- `build_timing_manifest` and `build_srt` receive only validated spans.
- Text-only Qwen never creates a synthetic full-audio segment.
- Existing timing provider defaults and existing SRT schema are unchanged.

Focused commands after implementation:

`uv run --offline pytest -q tests/test_asr_timing_bridge.py tests/test_cli_json_contract.py`

`uv run --offline pytest -q tests/test_generation_stability.py`

### S6 — Nemotron offline adapter, fixture-first

Purpose: add `nemotron-local` only after a pre-provisioned runtime boundary is
chosen and documented.

The adapter uses the generic registry, not the legacy timing selection chain.
It declares the actual executable/library probe, model-path/revision receipt,
supported devices/compute, and response capabilities. Phrase boosting, if the
installed runtime supports it, accepts phrases with only semantic strength
`mild`, `normal`, or `strong`; it must not expose invented portable numbers.

Acceptance:

- Offline fixture tests cover missing runtime, missing model, non-zero child
  status, malformed output, no-word response, and valid spans.
- The adapter has no cloud fallback and no API-key check.
- An optional future timing bridge is enabled only when S5 validation passes.
- Streaming is not represented as a `timings` or `transcribe` flag.

Focused command after implementation:

`uv run --offline pytest -q tests/test_nemotron_asr_provider.py tests/test_asr_registry.py tests/test_asr_timing_bridge.py`

### S7 — Corpus and benchmark harness

Purpose: validate manifest/schema and record reproducible results without
pretending that a single measurement proves product quality.

Planned files:

- `tests/fixtures/asr_corpus/manifest.json` or an equivalent approved
  repository-local manifest with no private data.
- `tests/test_asr_corpus_manifest.py`: pairing, SHA-256, required/forbidden
  anchor, state/language, and license-status validation.
- `docs/asr-benchmark-schema.md` only when the harness schema is implemented
  and covered; link it from `docs/README.md`.

Acceptance:

- Corpus cases include clean short speech, technical terms/code-switching,
  numbers/commands, silence/noise, degraded speech, and a separate
  hand-aligned sample for timing.
- Manifest pairs audio and UTF-8 references and records source/provenance,
  hashes, expected state/language, categories, and license status.
- WVM Slice 5 remains reference-only until an owner approves redistribution.
  Any approved subset preserves original OGG+TXT pairing and original SHA-256.
- Benchmark rows pin model/revision, runtime/version, device/compute,
  language, context/phrase-profile hashes, segmentation, timestamp mode,
  cold/warm state, WER/CER, term behavior, non-speech insertions, resource
  peaks, and alignment quality.
- Model benchmark, pipeline benchmark, and any future product/streaming
  benchmark are separate tables.

Focused command after implementation:

`uv run --offline pytest -q tests/test_asr_corpus_manifest.py`

## Test strategy

All development tests remain deterministic and offline:

1. Contract tests construct model types directly.
2. Registry tests use dummy factories and simulated missing imports.
3. CLI tests run through the existing JSON helper with temporary audio/output
   paths and mocked providers.
4. Adapter tests use sanitized response fixtures or mocked subprocess receipts.
5. Corpus tests validate schemas and hashes without executing inference.
6. A separately approved local experiment is the first runtime proof; it is
   recorded as `RUNTIME`, not relabeled as unit-test evidence.

At the end of a code slice, run the changed focused tests first and then
`uv run --offline pytest -q` when the implementation card's scope permits.
Report the actual command outcomes; do not record a permanent transient test
count in these documents.

The narrow suite runs first. Broader project tests follow only when the
implementation slice actually changes code. No dependency lockfile change is
made merely to execute tests.

## Migration and documentation gates

Before any public CLI slice is merged:

- compare old and new `timings` JSON/SRT output and exit behavior with existing
  tests;
- prove legacy parser choices, doctor checks, list output, and
  `faster-whisper` defaults are unchanged;
- add each new ID to registry-derived output only, avoiding synchronized
  hand-edited lists;
- document only verified command flags, IDs, JSON schema, and exit behavior in
  `docs/agent-cli-contract.md`;
- update `docs/README.md` and provider docs in the same slice;
- never document unverified model availability, price, accuracy, hardware
  compatibility, or streaming behavior as fact.

## Evidence status and unresolved questions

Static repository evidence supports this plan: current timing providers and
artifacts are segment-dependent, and parser/doctor/preflight/dispatch/list
registration is duplicated. It does not prove any candidate runtime.

Still unresolved before adapter implementation:

- exact supported package/runtime distribution and version constraints;
- model artifact/version/provenance and acquisition policy;
- actual Qwen/Nemotron response schema, timestamp reliability, and hardware
  behavior;
- canonical local corpus contents and WVM redistribution approval;
- product error budgets for accuracy, latency, memory, non-speech behavior,
  and alignment quality;
- a future streaming session/event API.

Resolve each item through the smallest approved experiment or owner decision;
do not turn this plan into a substitute for runtime evidence.