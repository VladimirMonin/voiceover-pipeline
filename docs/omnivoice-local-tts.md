# OmniVoice Local TTS

`omnivoice-local` is an explicit local, offline route for Russian and other supported text. It uses one fixed built-in female style condition, not a named voice ID, AutoVoice selection, cloning, or voice design. It is not a default provider and does not replace the existing cloud providers, Qwen local TTS, or Faster-Whisper timing path.

## Scope

The current route supports only fixed offline female-style synthesis:

- no server or persistent model process beyond the one container session for a run;
- built-in `female` condition only: no named preset/voice ID, AutoVoice selection, cloning, voice design, streaming, or public style controls;
- deterministic seed `1234`, 32 inference steps, guidance scale `2.0`, and internal text chunk size `420`;
- validated mono RIFF/WAVE output at the model's expected 24 kHz before VOP consumes it.

Passing a named `--voice`, Qwen cloning/sample options, and `--style-prompt` / `--style-prompt-file` to `omnivoice-local` is rejected. `--mode preset` is accepted only because it is the CLI parser's existing neutral default; it does not select a Qwen preset voice. Public state and chunk manifests record the built-in `female` condition and the single-session strategy.

## Long-form chunk policy

The explicit pair `omnivoice-local` + `audio-cpp/omnivoice-q8_0` sentence-packs
spoken text into atoms of at most 420 characters before synthesis. It preserves
token order, keeps short introductory sentences with their following sentence
where possible, and rejects raw digits so dates, versions, percentages, and
fractions must be written in words. This is an evidence-based OmniVoice profile:
cloud providers and unprofiled local provider/model pairs are not rewritten by
this policy. The prepared atoms are then joined into one request, and
`audiocpp_cli --text-chunk-size 420` performs the internal splits inside the
same container/model session. A bounded `--limit-chunks 3` run therefore creates
one session containing the first three prepared fragments, rather than three
independently initialized voices.

## Approved local inputs

| Item | Exact value |
|---|---|
| audio.cpp source | `502b5b74bd26e9b4aed267d1776ecf131cae7215` |
| Immutable CUDA image | `ghcr.io/0xshug0/audio.cpp@sha256:b46770ff33321ad187329659eb38ef22a5ae2bc6a8295f00a4f3b785b4211e58` |
| `audiocpp_cli` SHA-256 | `d98b99f10355a018ddaec6d17999725ab7bdbcf5f164ab067c1288a15a4f51dd` |
| GGUF artifact | `audio-cpp/omnivoice-q8_0` (`Q8_0 GGUF`) |
| GGUF SHA-256 | `2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b` |
| GGUF source revision | `audio-cpp/audio.cpp-gguf@c3857f1ec35cfea8993924e7c2a6f682b5dc060b` |

The upstream OmniVoice weights are CC-BY-NC-4.0. This route is for local, noncommercial research only: it must not bundle, redistribute, publish, or claim commercial availability for the model.

VOP requires an explicit local-use acknowledgment before it admits the artifact:

```bash
export VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE=accept-cc-by-nc-4.0-local-use
```

The configured file is streamed through SHA-256 verification before dependency health or provider construction succeeds. Its digest must equal the value above; an arbitrary existing GGUF is unavailable rather than a fallback model. This acknowledgment records local noncommercial use only. It is not a license grant and does not permit redistribution, publication, or commercial use.

## Linux configuration

The one required model file must be explicitly placed under local storage. VOP never downloads it.

```bash
export VOICEOVER_OMNIVOICE_MODEL=/media/v/storage/voiceover-pipeline/omnivoice/model/omnivoice-q8_0.gguf
export VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE=accept-cc-by-nc-4.0-local-use
export VOICEOVER_OMNIVOICE_CONTAINER_COMMAND_JSON='["sudo","-n","docker"]'
uv run voiceover doctor --provider omnivoice-local --json
uv run voiceover generate --provider omnivoice-local --script script.md --run-id omnivoice-test --limit-chunks 1 --json
```

`VOICEOVER_OMNIVOICE_CONTAINER_COMMAND_JSON` is optional and defaults to `["docker"]`; when set, it must be a JSON array of nonempty argv strings. The runtime starts a removable container with `shell=False`, `--network none`, a read-only root filesystem, read-only model mount, private 0700 output directory, `--gpus all`, and `--rm`.

`doctor` checks the explicit model-file/configuration boundary, the exact SHA-256, the noncommercial-use acknowledgment, and a local GPU probe. It does not download a model, run a model, or prove container execution; the first real request remains GPU-lease guarded.

## Platform boundary

On Linux the provider selects the pinned container route. On Windows it selects the existing native executable factory only when `VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE`, its adjacent checksummed EXE/DLL closure, the exact admitted model, and the noncommercial-use acknowledgment are all present; Docker and WSL are not fallback routes. This is static factory/package coverage only: native Windows inference, quality, resource, cancellation, and acceptance have not run and are not claimed.

## Receipts and safety

The runtime returns only copied WAV bytes and a runtime receipt containing driver ID, transport, pinned source revision, and `audiocpp_cli` build hash. Every generated OmniVoice chunk also preserves a public artifact receipt with model ID, SHA-256, quantization, license, and provenance; its public `voice_selection` says `built-in-style-condition` / `female`, and `voice_session` records the fixed seed and one-session internal chunking strategy. These fields contain no local model or temporary paths. Private temporary output paths and input text are not included in receipts or surfaced in transport errors. GPU leasing and lifecycle release are delegated to `LocalAudioRuntime`.
