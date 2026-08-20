# Pinned audio.cpp runtime boundary

## Scope and source pin

This repository has a runtime-neutral local model seam. The first native driver is
the `audio.cpp` spike candidate pinned to commit
`502b5b74bd26e9b4aed267d1776ecf131cae7215`. No binary, model weight, model
hash, or build receipt is committed here.

`python` remains the default local runtime. `audio-cpp` is explicit; `auto` can
choose it only for a family placed on the promotion allow-list. An unavailable
promoted driver falls back to `python`; an explicit `audio-cpp` choice fails
closed. This preserves current `qwen-local` and `nemotron-local` provider IDs
and their stored artifact contracts. Faster-Whisper and cloud providers remain
outside this runtime layer.

## Runtime wire boundary

`AudioCppRuntimeDriver` accepts and emits only schema-versioned JSON objects;
it never scrapes human-readable CLI output. Every envelope contains a request
ID, operation, family, provider ID, and generic payload. The provider-facing
contract therefore contains no `audio.cpp` command-line fields and can be
implemented by a future MLX driver without renaming providers or artifacts.

The subprocess transport runs without a shell, gives the child only a small
runtime-library/locale environment allow-list, creates a private per-request
temporary directory, bounds execution time, captures diagnostics without
placing them in public errors, and terminates the complete process group on
cancellation or timeout. On Windows it uses a new process group plus
CTRL_BREAK/terminate/kill escalation rather than POSIX session or `killpg`.
The shared GPU lease and live resource gates are not part of this foundation;
they are a later, separate lifecycle slice.

For a strict JSON TTS reply, the child may declare a WAV `response.audio_path`
inside that private workspace. Before cleanup, the subprocess transport resolves
and bounds that artifact, rejects missing, non-file, escaped or oversized paths,
then returns copied `audio_bytes` with `audio_format: "wav"`; it does not return
the temporary path to the typed provider response.

The native Windows CLI route accepts only an explicit `.exe` package with a
co-located `audio_cpp_dependency_closure.json`; every listed EXE/DLL file is
SHA-256 checked before provider selection. Its typed codec keeps VOP envelopes
independent of Docker/WSL paths and uses private output workspaces for text,
word, segment and WAV outputs. The code and fake-process tests cover launch,
cleanup and cancellation mechanics only: no Windows binary, model load, CUDA
probe or inference claim follows from this boundary.

## Build metadata boundary

Use `scripts/build_audio_cpp.py` only to emit a declarative plan. It neither
clones sources nor configures CMake nor downloads weights:

```text
uv run --offline python scripts/build_audio_cpp.py \
  --source /opt/src/audio.cpp --build-dir /opt/build/audio.cpp-cuda \
  --backend cuda --compiler clang++ \
  --cmake-definition CMAKE_CUDA_ARCHITECTURES=86 --emit-plan
```

A real build card must first verify the checkout's exact HEAD, clean source
tree, source-specific CMake flags, binary dependencies, CUDA architecture, and
binary SHA-256. Build output must be outside the source checkout and Git.

Inventory support is metadata-only: CPU, CUDA, HIP/ROCm, Vulkan, and Metal are
listed as the candidate's supported backends. MLX is explicitly
`not-installed-not-implemented`, not an `audio.cpp` backend.

## Family inventory and non-claims

| Family | Public provider ID | Planned timestamp origin | Prompt contract | State |
| --- | --- | --- | --- | --- |
| Qwen3 ASR | `qwen-local` | none before optional aligner | free contextual text | inventory-only |
| Qwen3 Forced Aligner | `qwen-local` | forced | alignment companion | inventory-only |
| Nemotron 3.5 ASR | `nemotron-local` | native | typed language/task dictionary; no phrase hints | inventory-only |
| Qwen3 TTS | `qwen-local` | none | typed voice and instruction fields | inventory-only |
| OmniVoice | `omnivoice-local` | none | typed voice and style fields | inventory-only |

The inventory records model identifiers and installation state, not fabricated
weights metadata: hash and quantization remain unset until an approved artifact
is installed and evidenced. No inference, model load, CUDA probe, or provider
promotion has been performed by this foundation.
