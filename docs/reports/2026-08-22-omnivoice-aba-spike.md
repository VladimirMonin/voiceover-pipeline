# Spike report: OmniVoice two-voice dialogue A/B/A capability

Date: 2026-08-22

## Purpose

Prove that three sequential single-voice native OmniVoice calls with different
voice-bank profiles produce audibly different voices, and that a first+third
call with the same profile (and the same fixed seed) produce a deterministic
identical clip. This is the capability evidence needed before implementing
two-voice dialogue support (one native call per turn, per-speaker
voice-bank profile).

Local, free, offline runtime only. No paid/cloud provider, no `.env` read,
no git operations.

## Environment and runtime

- Provider: `omnivoice-local` (native audio.cpp `audiocpp_cli.exe` on Windows).
- Model: `audio-cpp/omnivoice-q8_0` (GGUF, license CC-BY-NC-4.0, local
  noncommercial use acknowledged via env flag).
- Model artifact SHA-256 `2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b`
  matches the pinned inventory; native package admission passed.
- `voiceover doctor --provider omnivoice-local --json` -> `workflow_ok: true`
  with the explicit env setup used for every call below.
- Seed is NOT CLI-controllable: fixed `OMNIVOICE_DEFAULT_SEED = 1234`,
  `num_inference_steps = 32`, `guidance_scale = 2.0`, language `ru`
  (confirmed in the per-run JSON receipt `voice_session.seed: 1234`).
  Same-profile calls therefore always use the same seed, which makes the
  A1 == A2 determinism test meaningful by default.

## Profiles used

- `omni-female-neutral-01` (profile A)
- `omni-male-deep-01` (profile B)

Control text: short 1-2 sentence Russian phrase, one chunk (81 chars), file
`in/spike-aba-omnivoice-20260822/control-text.md` (untracked; kept outside
the report).

## Commands run (sequential, three native calls)

Every call used the same verified environment and CLI shape:

```powershell
$env:VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE="C:\audio-cpp-work\pkg\audiocpp_cli.exe"
$env:VOICEOVER_OMNIVOICE_MODEL="C:\audio-cpp-work\pkg\models\omnivoice\omnivoice-q8_0.gguf"
$env:VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE="accept-cc-by-nc-4.0-local-use"

uv run voiceover generate `
  --provider omnivoice-local `
  --mode preset `
  --voice-bank "C:\audio-cpp-work\voice-bank\approved\catalog.json" `
  --voice <PROFILE> `
  --script "in\spike-aba-omnivoice-20260822\control-text.md" `
  --output-dir "out\spike-aba-omnivoice-20260822" `
  --run-id <RUN_ID> `
  --json
```

- A1: `--voice omni-female-neutral-01 --run-id A1`
- B:  `--voice omni-male-deep-01    --run-id B`
- A2: `--voice omni-female-neutral-01 --run-id A2`

CLI emits MP3; each WAV was extracted with a deterministic lossless decode
`ffmpeg -y -loglevel error -i <run>.mp3 out\spike-aba-omnivoice-20260822\<RUN>.wav`
(no trimming). No credentials were involved; commands above are unredacted.

## Per-call table

| Call | Profile ID | seed | steps | guidance | Duration (audio) | Wall | Output WAV | SHA-256 (WAV) | Exit |
|---|---|---|---|---|---|---|---|---|---|
| A1 | omni-female-neutral-01 | 1234 | 32 | 2.0 | 5.16 s | 13.97 s | A1.wav | `4DBB3DBF0147E0CE61DADF8A99C97033902107B67EEAA43BC2619F41C6556C39` | 0 |
| B | omni-male-deep-01 | 1234 | 32 | 2.0 | 5.20 s | 13.27 s | B.wav | `F548C33479A934BA01261ACB616F465DA56DAE39CD91CA299AB3A634ED50A7D4` | 0 |
| A2 | omni-female-neutral-01 | 1234 | 32 | 2.0 | 5.16 s | 13.86 s | A2.wav | `4DBB3DBF0147E0CE61DADF8A99C97033902107B67EEAA43BC2619F41C6556C39` | 0 |

MP3 SHA-256 (identical decode source, extra evidence):
- A1: `6CFF4B37EE38EC512B7A74A3C1866B8A741CD033F40BF154107991AFED6A13C1`
- B:  `4FA606ECC4C2164E20D658719269FA062222A86C4AF731F491F159456C7A2860`
- A2: `6CFF4B37EE38EC512B7A74A3C1866B8A741CD033F40BF154107991AFED6A13C1`

Cost: 0.0 RUB per call (local model; no billing request).

## Catalog fingerprints

| profile | fingerprint (reference SHA-256) |
|---|---|
| `omni-female-neutral-01` | `e5eeb79bb4e75f425295fcb5b808f48164ac2d8f8d61aa7e2250ad3b14239433` |
| `omni-male-deep-01` | `6ebf66855d972a0c566aabd553d2884e37b8ef209105de48b6032c1ab3c52bcf` |

Distinct: yes (both values differ in the full digest; confirmable in the
bank catalog `schema_version: 1`). Both were also present in the per-run
`voice_selection.voice_fingerprint`.

Runtime receipt per run (all three identical):
`runtime_receipt.sha256 = 2f4be637…3da8b`, `license = CC-BY-NC-4.0 upstream weights; local noncommercial research only`,
`provenance = audio-cpp@502b5b74bd26e9b4aed267d1776ecf131cae7215 / OmniVoice-GGUF@c3857f1ec35cfea8993924e7c2a6f682b5dc060b`.

## Verdicts

(a) B audibly differs from A: **NOT confirmed — human listen required.**
Machine evidence is strongly supportive but not conclusive: different profile
(fingerprint `e5eeb79b…` vs `…`), different audio bytes
(B SHA differs from A1/A2), different duration (5.20 s vs 5.16 s). A human
listener must confirm the difference is audible.

(b) A1 == A2 deterministic: **PASS (machine evidence).**
    A1.wav and A2.wav SHA-256 identical
    (`4DBB3DBF0147B0CE61DADF8A99C97033902107B67EEAA43BC2619F41C6556C39`),
    byte-identical size (247 758 bytes), identical MP3 hashes, identical
    duration (5.16 s). Seed is fixed (1234), not CLI-controllable, so
    determinism holds by construction of the runtime; the observed byte
    identity is the evidence.

## Output directory

Relative to repo root (`out/` is gitignored; not committed):

```
out/spike-aba-omnivoice-20260822/
├── A1.wav
├── B.wav
├── A2.wav
├── A1/  (run artifacts: manifest.json, run_state.json, generation.log,
│        chunks/chunk_01_omnivoice_session.mp3, A1-voiceover-….mp3 + .json)
├── B/   (same layout)
└── A2/  (same layout)
```

Source script (untracked): `in/spike-aba-omnivoice-20260822/control-text.md`

## Blockers and notes

- Initial attempt failed: GPU was busy (238 MiB free of 16.3 GiB;
  `insufficient free GPU memory for voiceover job`, lease threshold 4 GiB).
  After freeing VRAM (~14.9 GiB free) all three calls succeeded.
- First A1 folder had been created by a failed attempt; removed before the
  successful run (no `--overwrite` needed).
- The CLI exposes no `--seed` flag; fixed seed 1234 is used and is
  sufficient for the determinism claim.
