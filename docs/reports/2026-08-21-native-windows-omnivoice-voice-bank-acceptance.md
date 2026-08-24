# Acceptance: native Windows OmniVoice voice bank (NW-10 / NW-11)

Date: 2026-08-21

Scope: local `omnivoice-local` provider on Windows via audio.cpp native CLI,
four approved design voices in a local JSON voice bank, and the
`auto / preset / clone / design` mode matrix.

All runs used the pinned native runtime and the local OmniVoice GGUF model.
Model weights, audio files, prompts, transcripts, and absolute paths stay
outside Git.

## Artifacts

- Native executable: `C:\audio-cpp-work\pkg\audiocpp_cli.exe`
  (audio.cpp revision `502b5b74bd26e9b4aed267d1776ecf131cae7215`)
- Model: `omnivoice-q8_0.gguf`, model SHA-256
  `2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b`,
  license CC-BY-NC-4.0, local noncommercial use accepted by environment flag
- Nemotron ASR (back-transcription checks): `nemotron-3.5-asr-streaming-0.6b-q8_0.gguf`
- Voice bank: `C:\audio-cpp-work\voice-bank\approved\catalog.json` (schema v1,
  outside Git)

## Approved voice profiles

| id | language | reference SHA-256 |
|---|---|---|
| `omni-female-neutral-01` | ru | `e5eeb79bb4e75f425295fcb5b808f48164ac2d8f8d61aa7e2250ad3b14239433` |
| `omni-female-deep-01` | ru | `539ec8b8b556452ae6c1eca71c506dab92532131671a3ce7d33b3756b1a6d194` |
| `omni-male-neutral-01` | ru | `f9e0f681916f405216fc0f78c0d6711d0d10e5afc8cdd54d4d6d24758c388e92` |
| `omni-male-deep-01` | ru | `6ebf66855d972a0c566aabd553d2884e37b8ef209105de48b6032c1ab3c52bcf` |

Origin: voice design (gender, age, pitch instruction), seed `1234`,
`--language ru`, `--num-inference-steps 32`, `--guidance-scale 2.0`,
24 kHz PCM16 mono, ~4.0 s each.

## Live acceptance results

### UTF-8 root-cause fix

The earlier noisy candidates were produced by an invocation bug: the text
file was read through the Windows PowerShell 5.1 default codepage instead of
UTF-8, sending mojibake characters to the model; `--language ru` was also
missing. A strict UTF-8 preflight (code-point check on the first four
characters) now guards every direct invocation. A short corrected smoke run
produced natural Russian speech.

### Determinism and A/B/A

- Profile A (`omni-female-neutral-01`, text X) run twice:
  output MP3 SHA-256 identical (`14232d1d2f72b2ae…`), duration identical (6.64 s).
- Profile B (`omni-male-deep-01`, text X) differs from A and carries a
  different voice fingerprint.
- Profile A on text Y differs in content; fingerprint unchanged.
- No reference state is shared between runs; each run is a fresh process.

### Voice bank admission

`list voices --json` returns exactly the four approved ids; public profiles
expose only id, display_name, description, language. No transcript, absolute
path, or reference instruction is published. Catalog SHA-256 digests match the
files on disk; mono-WAV and containment checks pass.

### Preset run through the bank

`--mode preset --voice omni-female-neutral-01 --voice-bank …` completed with
exit 0:

- metadata kind `bank-preset`, voice id and fingerprint present;
- runtime strategy `bank-preset-native-session`, seed `1234`;
- no `reference_text` / `reference_audio` / bank-path leaks in JSON or log;
- resume re-run reused the completed run without regeneration
  (MP3 hash unchanged, exit 0);
- a resume against a tampered fingerprint was rejected with exit code 30.

### Clone mode (ad-hoc reference)

`--mode clone` with a bank reference WAV + transcript completed with exit 0:

- metadata kind `reference-clone`, strategy `reference-isolated-native-session`;
- no bank path or reference transcript in JSON;
- resume with the same reference completed with exit 0 (identity is
  deterministic across processes — builtin `hash()` replaced by SHA-256).

### Long form

- Single-section script of 36.78 s: one native session, internal chunking,
  one voice fingerprint across the whole audio, no reference leaks.
- Three-section script (delimiter-separated): sections merged into one native
  session by design, all sections present, fingerprint preserved.

### Quality and cleanup

- All approved voices verified by Nemotron back-transcription: exact match
  (62/62 characters) with the intended text.
- Final MP3: 128 kbps libmp3lame via ffmpeg concat demuxer.
- No lingering `audiocpp_cli.exe` / ffmpeg processes after runs; staging temp
  workspaces cleaned; no JSON or log leaks of reference material.

## Offline gates

- Focused suites: 280 passed (incl. new clone-identity determinism test).
- Full suite: 626 passed, 3 skipped, 37 failed — identical set to the
  pre-existing Windows baseline (container/qwen-asr/wvm, all outside the
  touched files).
- `ruff check` / `ruff format --check`: clean.
- `mypy`: 51 pre-existing errors in 4 transport files, 0 new.

## Limitations

- CC-BY-NC-4.0 model: local noncommercial research use only.
- Voice design is most stable for the trained languages. The short Russian
  references captured into this accepted voice bank were validated by listening
  and back-transcription on this seed; this does not support long Russian Voice
  Design or claim that hallucination is repaired. Runtime use of these accepted
  profiles follows the preset/clone path.
- Direct PowerShell invocations must pass strict UTF-8 text (see
  `docs/omnivoice-local-tts.md`) and `--language ru` for Russian.
