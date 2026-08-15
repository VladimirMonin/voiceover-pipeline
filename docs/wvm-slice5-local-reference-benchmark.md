# WVM Slice 5 local-reference ASR benchmark

`tests/fixtures/wvm_slice5_benchmark/manifest.json` is a deterministic, local-reference corpus manifest for exactly 50 owner-approved synthetic WVM Slice 5 cases. It records the canonical case ID, tier, category, state, language, anchors, pause/noise metadata, WVM-root-relative OGG/TXT paths, expected reference text, and SHA-256 values for both files.

## Local reference only

No WVM audio binary, WVM source code, or WVM import is present in this repository. The manifest intentionally contains no absolute local path. It can be used only when a local WVM checkout is supplied explicitly as a corpus root.

Set `WVM_ROOT` outside this repository to the WVM checkout root, then pass it to the developer benchmark tool as `--corpus-root "$WVM_ROOT"`. The expected asset paths resolve under that root, for example `tests/assets/whisper/slice5/lang_en_clean_first5.ogg`.

The tool never discovers a WVM checkout automatically. Without `--corpus-root`, it treats the manifest directory as the root and fails before an adapter call when the referenced audio is missing. A supplied non-directory root fails with `Benchmark corpus root not found`. Both behaviors are deliberate fail-closed missing-root handling.

## Offline integrity validation

The focused validation reads the local external pair files and checks all 50 audio/reference hashes plus expected text. It does not load a model, run inference, use a GPU, download anything, or call a provider:

```bash
WVM_ROOT=/path/to/Whisper-Voice-Machine \
  uv run --offline pytest -q tests/test_wvm_slice5_benchmark_manifest.py
```

The general developer benchmark tool accepts the same root argument. Running an adapter is a separate resource-gated operation and may load a local model; do not treat this documentation as approval to perform that operation.

## Metadata and metric boundary

The selection uses fixed six- or seven-case quotas for each canonical WVM category and retains P0/P1/P2 coverage. `state` remains authoritative for `no_speech`, `filtered_or_rejected`, `non_empty`, and `diagnostic` scenarios. For no-speech cases, `expected_text` faithfully mirrors the paired canonical TXT descriptor; it is not a claim that the descriptor words were spoken.

The generic harness currently reports normalized text metrics and does not turn the WVM `state` field into a scored quality verdict. Do not use a WER/CER value across this manifest as a model-quality claim until a state-aware scoring policy is explicitly added and validated.

## Privacy and redistribution boundary

The owner approved local Voiceover Pipeline evaluation use for this selected synthetic WVM material. The audio and references remain `NOASSERTION`: that approval is not a redistribution, publication, or release license. Do not copy audio files, package them, publish them, or treat this manifest as a redistribution grant. A future release or distribution of a corpus based on these assets requires separate explicit provenance and licensing approval.
