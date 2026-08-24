# OmniVoice Local TTS

`omnivoice-local` is an explicit local, offline route for Russian and other supported text. It supports four synthesis modes: `auto`, `preset` (local voice bank → clone), `clone` (ad-hoc reference), and `design` (instruction allowlist). On Windows it runs through the native `audiocpp_cli.exe` factory; on Linux it runs through the pinned immutable container. It is not a default provider and does not replace the existing cloud providers, Qwen local TTS, or Faster-Whisper timing path.

## Modes

| Mode | Behaviour | Public `voice_selection.kind` | Session strategy |
|---|---|---|---|
| `auto` | No voice guidance; model default | `auto-voice` | `auto-voice-native-session` |
| `preset` (default) | Resolves `--voice` through `--voice-bank` catalog into a reference WAV + transcript, then native clone | `bank-preset` (with `voice_id`, `voice_fingerprint`) | `bank-preset-native-session` |
| `clone` | Ad-hoc `--reference-audio` + `--reference-text` native clone | `reference-clone` | `reference-isolated-native-session` |
| `design` | Allowlisted `--design-instruction` (gender, age, pitch, style; accents are English-only) | `design-instruction` | `design-instruction-native-session` |
| (legacy fallback) | Fixed built-in `female` condition, only when no mode applies | `built-in-style-condition` | `single-native-invocation-internal-text-chunking` |

Deterministic seed `1234`, 32 inference steps, guidance scale `2.0`, and
internal text chunk size `420` apply to every mode.

`--voice` is rejected for `auto`, `clone`, and `design`; `preset` requires a
voice bank catalog and rejects unknown voice ids with exit code 30 on resume
mismatch. Passing Qwen cloning/sample options and `--style-prompt` /
`--style-prompt-file` to `omnivoice-local` is rejected.

Upstream trains Voice Design only on Chinese and English. Accent attributes
describe English speech, not the language of synthesis. Because the current VOP
OmniVoice route is fixed to Russian, accent and Chinese-dialect attributes are
rejected before runtime admission. Russian Voice Design is an unsupported model
regime, not a repaired hallucination route: short requests at or below the
estimated 30-second threshold remain explicitly experimental with a warning;
longer requests fail before provider/model/GPU construction. A syntactically
valid instruction such as `female, middle-aged, very low pitch` does not make
long Russian Voice Design reliable.

## Voice bank

A voice bank is a directory outside the repository with `catalog.json`
(schema v1) and a `voices/` subdirectory:

```bash
uv run voiceover-pipeline list voices --provider omnivoice-local \
  --voice-bank "C:\path\to\approved\catalog.json" --json
uv run voiceover-pipeline generate --provider omnivoice-local --mode preset \
  --voice omni-female-neutral-01 \
  --voice-bank "C:\path\to\approved\catalog.json" \
  --script script.md --output-dir out --json
```

Catalog invariants (enforced by the loader):

- reference paths are relative to the bank root; absolute paths, drive
  letters, and `..` escapes are rejected;
- each profile carries `reference_sha256` (64-hex), verified against the file
  before every run;
- reference audio must be a readable mono WAV (any sample rate; the transport
  normalizes it to PCM16 mono 24 kHz during staging);
- public metadata exposes only id, display name, description, language, and
  fingerprint — never transcripts, absolute paths, or instructions.

Approved local bank: `C:\audio-cpp-work\voice-bank\approved\catalog.json`
(four Russian design voices, seed 1234; acceptance in
`docs/reports/2026-08-21-native-windows-omnivoice-voice-bank-acceptance.md`).

## Long-form chunk policy

Before the runtime policy below applies, VOP estimates duration offline at a
conservative 2.5 words/second. For `design` in languages other than English or
Chinese, an estimate above the upstream 30-second long-form threshold is rejected
with exit code 2. JSON errors include `details.error_code`, the estimate and
threshold, `automatic_fallback: false`, and four explicit alternatives: Russian
reference cloning, an available accepted voice-bank preset, separately accepted
short design clips (experimental), or another TTS provider. VOP never silently
changes mode or voice. Clone, preset/voice-bank and native Windows routes are
not rejected. The policy permits English/Chinese design, but the current CLI
route remains fixed to Russian and does not expose a language selector.

The explicit pair `omnivoice-local` + `audio-cpp/omnivoice-q8_0` sentence-packs
spoken text into atoms of at most 420 characters before synthesis. It preserves
token order, keeps short introductory sentences with their following sentence
where possible, and rejects raw digits so dates, versions, percentages, and
fractions must be written in words. This is an evidence-based OmniVoice profile:
cloud providers and unprofiled local provider/model pairs are not rewritten by
this policy. The prepared fragments are then joined into one native session,
and `audiocpp_cli --text-chunk-size 420` performs the internal splits inside
the same model session, so the voice (and seed) stays constant across the whole
script. A bounded `--limit-chunks 3` run therefore creates one session
containing the first three prepared fragments, rather than three
independently initialized voices.

For a session longer than the internal limit, the completed prepared atoms are
whitespace-padded to the next 420-character boundary, so each internal split
lands after its terminal punctuation rather than inside a spoken word. An atom
that exactly fills the limit already ends on that boundary and needs no padding.
`inspect_omnivoice_internal_seams` is the deterministic diagnostic for this
invariant; a rare overlong sentence that can only be split between words is
reported as word-safe rather than falsely described as sentence-safe.

The Linux container transport uses the 300-second base timeout per 420-character
workload unit, scaling with the decoded request text and capping the total wait
at 1,800 seconds. On timeout or cancellation it terminates and reaps the whole
container process group before the private workspace is removed; successful WAV
bytes are decoded into the response before that cleanup. Retrying therefore
cannot begin while the previous container process is still being reaped. The
Windows native factory and its platform-specific process semantics are unchanged.

## Approved local inputs

| Item | Exact value |
|---|---|
| audio.cpp source | `502b5b74bd26e9b4aed267d1776ecf131cae7215` |
| `audiocpp_cli` (Windows native) | `C:\audio-cpp-work\pkg\audiocpp_cli.exe` |
| GGUF artifact | `audio-cpp/omnivoice-q8_0` (`Q8_0 GGUF`) |
| GGUF SHA-256 | `2f4be637278043c6842de5b85d681532030e9eb6ffe0f8b0e320f68238e3da8b` |
| GGUF source revision | `audio-cpp/audio.cpp-gguf@c3857f1ec35cfea8993924e7c2a6f682b5dc060b` |

The upstream OmniVoice weights are CC-BY-NC-4.0. This route is for local,
noncommercial research only: it must not bundle, redistribute, publish, or
claim commercial availability for the model.

VOP requires an explicit local-use acknowledgment before it admits the artifact:

```bash
export VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE=accept-cc-by-nc-4.0-local-use
```

The configured file is streamed through SHA-256 verification before dependency
health or provider construction succeeds. Its digest must equal the value
above; an arbitrary existing GGUF is unavailable rather than a fallback model.
This acknowledgment records local noncommercial use only. It is not a license
grant and does not permit redistribution, publication, or commercial use.

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

## Windows native configuration

On Windows the provider selects the native executable factory when
`VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE`, its adjacent checksummed EXE/DLL
closure, the exact admitted model, and the noncommercial-use acknowledgment
are all present; Docker and WSL are not fallback routes.

```powershell
$env:VOICEOVER_AUDIO_CPP_NATIVE_EXECUTABLE = "C:\audio-cpp-work\pkg\audiocpp_cli.exe"
$env:VOICEOVER_OMNIVOICE_MODEL = "C:\audio-cpp-work\pkg\models\omnivoice\omnivoice-q8_0.gguf"
$env:VOICEOVER_OMNIVOICE_NONCOMMERCIAL_LOCAL_USE = "accept-cc-by-nc-4.0-local-use"
uv run voiceover-pipeline generate --provider omnivoice-local --mode preset `
  --voice omni-female-neutral-01 `
  --voice-bank "C:\audio-cpp-work\voice-bank\approved\catalog.json" `
  --script script.md --output-dir out --json
```

## Direct native invocation (UTF-8 rules)

Direct `audiocpp_cli.exe` calls must pass Russian text as proper UTF-8 and set
the language explicitly. Windows PowerShell 5.1 reads UTF-8 files without BOM
as the ANSI codepage, which corrupts Cyrillic (e.g. `Всем` → mojibake) and
yields noisy, unusable speech. Use strict byte reads and verify the codepoints
before invoking:

```powershell
$utf8 = [System.Text.UTF8Encoding]::new($false, $true)
$text = [System.IO.File]::ReadAllText("script.txt", $utf8).Trim()
# optional guard: the first four chars of "Всем" are 1042,1089,1077,1084
& "C:\audio-cpp-work\pkg\audiocpp_cli.exe" --task tts --family omnivoice `
  --model "C:\audio-cpp-work\pkg\models\omnivoice\omnivoice-q8_0.gguf" `
  --backend cuda --language ru --text $text `
  --instruct "female, young adult, moderate pitch" `
  --seed 1234 --num-inference-steps 32 --guidance-scale 2.0 --out out.wav
```

Always pass `--language ru` for Russian text. Never pipe text through
`Get-Content -Raw` without a `-Encoding UTF8` read.

## Receipts and safety

The runtime returns only copied WAV bytes and a runtime receipt containing driver ID, transport, pinned source revision, and `audiocpp_cli` build hash. Every generated OmniVoice chunk also preserves a public artifact receipt with model ID, SHA-256, quantization, license, and provenance; its public `voice_selection` records the mode-specific kind, and `voice_session` records the fixed seed and one-session internal chunking strategy. Run receipts additionally carry a path-free `execution_source` with package version, editable/wheel source kind, Git revision/dirty state where available, and exact package-tree SHA-256. Private paths, input text and design instructions are excluded. Use `voiceover verify-tts --audio ... --expected-file ... --provider ... --receipt ... --json` for transcript-free omission/insertion/repetition evidence; technical PASS never replaces human listening. Bank references are staged under a neutral filename, normalized to PCM16 mono 24 kHz, and cleaned up in `finally`. GPU leasing and lifecycle release are delegated to `LocalAudioRuntime`.
