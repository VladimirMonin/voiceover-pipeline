# audio.cpp Qwen ASR container recipe

This is the deterministic, per-request CLI route for Qwen3 ASR. VOP does not keep
a background audio.cpp server or a model process: each request is a removable,
cancelable container. This keeps GPU leasing, cancellation, and unload semantics
local to `LocalAudioRuntime`.

## Verified inputs

| Item | Pinned value |
|---|---|
| audio.cpp source | `502b5b74bd26e9b4aed267d1776ecf131cae7215` |
| CUDA image tag | `ghcr.io/0xshug0/audio.cpp:full-cuda13-20260816-502b5b7` |
| Image ID and immutable image reference | `sha256:b46770ff33321ad187329659eb38ef22a5ae2bc6a8295f00a4f3b785b4211e58` |
| Qwen3 ASR GGUF SHA-256 | `6c44ec2fb4cee513892d7863c1fcc3ea6b699ffa4d899b0ef4ab19956d9544f7` |
| Qwen3 ForcedAligner GGUF SHA-256 | `75209490b11cec2b0db749ca5f4ff92266f58efd30f7fd04d9eb2a3ac9cc929f` |

Before enabling this route, inspect the already-local image and model bytes. This
is a local verification step, not an inference run:

```bash
sudo -n docker image inspect \
  ghcr.io/0xshug0/audio.cpp:full-cuda13-20260816-502b5b7 \
  --format '{{.Id}}{{range .RepoDigests}}{{println}}{{.}}{{end}}'
sha256sum \
  /media/v/storage/audio.cpp/models/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf \
  /media/v/storage/audio.cpp/models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf
```

The inspection must show the table values. `sudo -n` must never prompt for or
record a password. The runtime itself uses the immutable `@sha256` reference,
not the mutable tag.

## VOP configuration

```bash
export VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON='["sudo","-n","docker"]'
export VOICEOVER_AUDIO_CPP_CONTAINER_IMAGE='ghcr.io/0xshug0/audio.cpp@sha256:b46770ff33321ad187329659eb38ef22a5ae2bc6a8295f00a4f3b785b4211e58'
export VOICEOVER_AUDIO_CPP_QWEN_ASR_MODEL='/media/v/storage/audio.cpp/models/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf'
export VOICEOVER_AUDIO_CPP_QWEN_FORCED_ALIGNER_MODEL='/media/v/storage/audio.cpp/models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf'
uv run voiceover doctor --with-asr --asr-provider qwen-local --asr-device cuda --asr-compute auto --json
```

When these variables are present, provider `qwen-local` selects the container
adapter. The legacy `VOICEOVER_AUDIO_CPP_BINARY` JSON-driver route remains
available and is only selected when the container image variable is absent.

`VOICEOVER_AUDIO_CPP_CONTAINER_COMMAND_JSON` is optional and defaults to the
single argv item `["docker"]`. It must be a JSON array of non-empty argv
strings; VOP passes that array directly to `subprocess.Popen` without a shell.
On this reviewed host the required prefix is exactly `["sudo", "-n", "docker"]`:
do not substitute a single shell-like string such as `"sudo -n docker"`.

## Exact request mapping and isolation

The adapter accepts only VOP schema version 1, operation `asr`, family
`qwen3-asr`, and model `Qwen/Qwen3-ASR-0.6B`. It stages input audio as private
16 kHz mono PCM WAV, then invokes the image with:

- `--task asr --family qwen3_asr --model /models/qwen3-asr.gguf --backend cuda`;
- `--language` and `--text` only when VOP supplied them;
- for `timestamp_mode=word`, `--session-option qwen3_asr.forced_aligner_model_path=/models/qwen3-forced-aligner.gguf` and `--words-out /output/words.json`.

The source audio, ASR GGUF and forced-aligner GGUF are individual read-only bind
mounts. The only writable mount is a mode-`0700` temporary output directory.
The container has `--network none`, a read-only root filesystem, a private
`/tmp`, and `--rm`. It emits its native transcript and word JSON only into that
temporary directory; the adapter validates and converts it to VOP JSON before
returning it. A cancellation or close terminates the active container process;
there is no persistent server model left to unload.

A future explicit server transport must prove the same request/response schema,
model identity, private-file boundary, cancellation, and GPU lifecycle before
it can replace this CLI route. This recipe does not claim a server adapter or a
successful inference run.

## Intended finite-audio invocation

After the owner has approved a GPU inference run, the public CLI surface is:

```bash
uv run voiceover transcribe \
  --audio 'recording.wav' \
  --provider qwen-local \
  --model 'Qwen/Qwen3-ASR-0.6B' \
  --language ru \
  --device cuda \
  --compute auto \
  --word-timestamps \
  --json
```

This task did not execute that command, load a model, or perform inference.

## Benchmark lanes

The 50-case WVM manifest is a transcript, resource, no-speech, and phrase-timing
lane. Its cases intentionally do not supply manual speech-window or word-boundary
truth, so boundary MAE/p95 is reported as `NOT_APPLICABLE` rather than causing a
run to abort or becoming zero. On Qwen, the existing forced-word response can
serve as a phrase envelope; word fields remain optional diagnostic evidence.

After separate owner approval for inference, run Python Qwen and audio.cpp Qwen
as separate invocations against the same local root, requiring finite,
non-negative, monotonic phrase intervals within each audio duration:

```bash
uv run --offline python tools/run_asr_benchmark.py \
  --corpus tests/fixtures/wvm_slice5_benchmark/manifest.json \
  --corpus-root /home/v/code/Whisper-Voice-Machine \
  --output-dir out/asr-benchmarks/qwen-python-50 \
  --provider qwen-local --device cuda --compute auto \
  --require-phrase-timing
```

`qwen-local` is the canonical Qwen family ID: with no audio.cpp runtime
environment it selects the Python adapter; with the configured audio.cpp
container environment it selects the audio.cpp adapter. Run those two states
separately with different `--output-dir` values; compare the recorded
`execution.runtime` receipt rather than treating them as one run.

The two-case `tests/fixtures/qwen_manual_russian_timing/manifest.json` is a
small optional diagnostic for word-boundary MAE/p95, coverage, monotonicity,
bounds, and zero-duration behavior. It is not a prerequisite for 50-case Qwen
acceptance. No >=180-second constructed input is bundled: the long chunk-offset
check is deferred until a safe existing artifact is available, and must not
delay the minimum parity run.
