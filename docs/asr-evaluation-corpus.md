# Synthetic ASR evaluation corpus

`tests/fixtures/asr_evaluation/` is a small offline corpus for adapter and benchmark work. Its canonical contract is `manifest.json`; every case has a SHA-256, expected text, language, category, pause profile, noise profile, and an expected speech window for future timestamp-boundary evaluation.

## Privacy and provenance

No WVM asset is copied or referenced by the corpus. The corpus is not derived from the WVM Slice5 inventory: its metadata states `wvm_assets_used: false` and it contains no WVM audio, transcripts, hashes, paths, or case identifiers.

Every WAV is generated locally from project-owned fixture text with FFmpeg's built-in `flite` filter (`voice=kal`), then optionally receives deterministic seeded white noise from the generator. It contains no private recording, personal conversation, customer content, uploaded media, cloud-provider response, model download, GPU inference, or network request.

The metadata and generator follow the repository's MIT license. Audio is marked **NOASSERTION**: the FFmpeg/flite voice-output redistribution terms have not been independently audited for a redistributable fixture release. Do not relabel the audio as MIT, public-domain, or WVM-approved without that legal/provenance review.

## Regeneration and integrity

Generation requires a local `ffmpeg` with the `flite` filter. It writes only inside the selected output directory and uses no provider credentials or network access:

```bash
uv run --offline python tools/generate_asr_evaluation_corpus.py
uv run --offline python tools/generate_asr_evaluation_corpus.py --check
```

The generator keeps the case order fixed, writes canonical pretty JSON (`sort_keys=True`), generates 16 kHz mono PCM WAV, and records a SHA-256 per audio file. `--check` verifies those hashes and WAV format without rendering speech. The loading tests in `tests/test_asr_evaluation_corpus.py` additionally verify the schema, case order, metadata, hashes, and speech-window bounds.

## Scope boundary

This corpus is a fixture source, not a benchmark result. It deliberately does not claim model accuracy, WER/CER, real-world noise robustness, timing quality, or equality with any provider output. Future harnesses may consume its expected text and speech-window metadata but must report generated measurements separately.
