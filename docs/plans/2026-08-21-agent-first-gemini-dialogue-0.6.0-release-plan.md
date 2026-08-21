# Agent-first release plan: Gemini dialogue and OmniVoice workflows (0.6.0)

> **Status:** approved for planning; implementation has not started.
> This plan targets the smallest safe `0.6.0` release for agent users.
> It does not authorize a paid provider call, package publication, release tag,
> or registry upload. Those operations require separate approval at their gates.

## Executive decision

The release does not need a generic multi-provider dialogue engine or external
Python orchestration scripts.

The repository already has a first-class two-speaker path:

- script format: `gemini-dialogue`;
- provider: `openrouter-tts`;
- model: `google/gemini-3.1-flash-tts-preview` through OpenRouter;
- request shape: Gemini `multi_speaker_voice_config`;
- output: normal resumable VOP artifacts.

Official Google GenAI types require exactly two `SpeakerVoiceConfig` entries.
The speaker name in the configuration must match the speaker name used in the
prompt. This matches the product shape already present in the repository.

The agent is the script author. A separate LLM-in-the-CLI feature for turning a
topic into a podcast is not required: when the user asks for a complete podcast,
the agent writes the structured dialogue script, validates it, and invokes the
existing synthesis command.

Local OmniVoice remains a first-class single-speaker-per-session path in this
release. A local two-speaker podcast may be prototyped outside the product, but
it is not a supported `0.6.0` workflow and must not require users or agents to
maintain helper Python scripts.

## Release outcome

An agent receiving a request such as "make a Russian technical podcast with a
male and female host" must be able to:

1. Load the `voiceover-pipeline` skill.
2. Author a valid `gemini-dialogue` Markdown file.
3. Validate it without a paid request.
4. Generate the two-speaker podcast with one VOP command.
5. Resume safely after an interruption without changing the cast.
6. Read one JSON response and find the artifacts through the normal manifest.

The canonical execution surface is:

```powershell
voiceover validate `
  --script "podcast.md" `
  --format gemini-dialogue `
  --agent `
  --json

voiceover generate `
  --script "podcast.md" `
  --run-id "podcast-prod" `
  --json `
  --resume
```

Provider, model, cast, performance direction, and dialogue format come from the
script frontmatter. Agents may pass explicit flags, but the self-contained
script is the preferred workflow.

## Canonical dialogue input

```markdown
---
format: gemini-dialogue
language: ru
model: google/gemini-3.1-flash-tts-preview
speakers:
  Host:
    display_name: Ведущая
    voice: Kore
    profile: warm, confident technical host
  Guest:
    display_name: Гость
    voice: Puck
    profile: calm, thoughtful technical expert
vibe: >
  Russian technical podcast. Natural question-and-answer conversation.
allowed_tags:
  - warmly
  - curious
  - serious
  - short pause
max_chunk_bytes: 3500
---

Host: [warmly] Что умеет утилита?
Guest: Она создаёт озвучку, тайминги и субтитры.

******

Host: [curious] Можно работать полностью локально?
Guest: Да. Для одноголосой озвучки доступны OmniVoice и Qwen.
```

## Fixed release boundaries

### Included

- Harden the existing Gemini two-speaker implementation.
- Make resume cast-safe.
- Keep `--json` safe for agents.
- Add one mocked end-to-end dialogue generation test.
- Synchronize the machine contract and distributable skill.
- Expose current OmniVoice `auto`, bank `preset`, ad-hoc `clone`, and `design`
  workflows to agents.
- Align project, runtime, skill, and changelog versions at `0.6.0`.
- Build and inspect local wheel/sdist artifacts.
- Perform one short live Gemini two-speaker smoke only after explicit paid-call
  approval.

### Explicitly excluded

- Direct Google GenAI SDK integration.
- NotebookLM-style source ingestion or automatic research.
- A generic multi-speaker abstraction shared by all TTS providers.
- Local multi-speaker OmniVoice or Qwen orchestration.
- Per-turn WAV files and per-turn timing metadata.
- A public voice-bank creation or mutation command.
- Installing CUDA drivers or downloading local model weights.
- Publishing a package, creating a tag, or creating a GitHub release without a
  separate approval.

## Current implementation inventory

The following product seams already exist and should be extended, not replaced:

| Area | Existing owner |
|---|---|
| Gemini dialogue parsing and validation | `src/voiceover_pipeline/gemini_dialogue.py` |
| CLI format/provider wiring | `src/voiceover_pipeline/cli.py` |
| OpenRouter multi-speaker request | `src/voiceover_pipeline/providers/openrouter_tts.py` |
| Resume state | `src/voiceover_pipeline/run_state.py` |
| Artifact manifests | `src/voiceover_pipeline/artifacts.py` |
| Gemini validation tests | `tests/test_cli_validation.py` |
| Provider payload tests | `tests/test_new_providers.py` |
| Resume tests | `tests/test_generation_stability.py` |
| JSON contract tests | `tests/test_cli_json_contract.py` |
| Machine-facing contract | `docs/agent-cli-contract.md` |
| Distributable agent skill | `docs/skills/voiceover-pipeline/SKILL.md` |
| Dialogue input guide | `docs/skills/voiceover-pipeline/docs/04-input-format.md` |
| Workflows | `docs/skills/voiceover-pipeline/docs/08-workflows.md` |
| Gemini prompting templates | `docs/skills/voiceover-pipeline/docs/12-gemini-prompting-templates.md` |
| Local runtime guidance | `docs/skills/voiceover-pipeline/docs/14-local-audio-cpp-models.md` |

## Release blockers

| ID | Problem | Why it blocks an agent-first release | Minimum resolution |
|---|---|---|---|
| GD-01 | Validator accepts one speaker and only rejects more than two | Official Gemini multi-speaker config requires exactly two entries | Require exactly two aliases |
| GD-02 | Two aliases may select the same voice | Product promises two distinct speakers | Require two distinct valid Gemini voices |
| GD-03 | Explicit top-level `--voice` can conflict with the cast | OpenRouter still needs a top-level voice, but it is compatibility metadata, not a third cast decision | Derive it from the first mapped speaker; reject a conflicting explicit value |
| GD-04 | Resume identity does not include cast, model, or style | A resumed paid run can mix audio generated with different voices or direction | Persist and compare a canonical dialogue identity |
| GD-05 | Some dialogue errors/diagnostics can pollute stdout | Agents require one parseable JSON object | Standard JSON error envelope; diagnostics to stderr |
| GD-06 | Main agent workflow does not route podcast requests to the existing dialogue path | Agents may bypass VOP and write helper scripts | Add the canonical branch and command |
| GD-07 | Skill says it does not write scripts even when asked for a complete podcast | An agent cannot complete the user's artifact request | Permit structured script authoring when the user requests the complete podcast/voiceover |
| GD-08 | Skill's OmniVoice guidance is stale | Agents cannot discover the shipped voice bank, clone, and design modes | Synchronize the minimal OmniVoice workflows |
| GD-09 | Skill source exists but was not registered in the current agent session | Correct instructions cannot trigger if unavailable | Document and verify installation through the agent platform's existing skill mechanism |
| GD-10 | Version metadata disagrees (`pyproject.toml` 0.5.1, package `__version__` 0.1.0) | Release and diagnostics report inconsistent versions | Align all version surfaces at 0.6.0 |
| GD-11 | Skill recommends a destructive standalone timings sequence | An agent can delete an existing generated run with `--overwrite` | Prefer `generate --with-timings`; otherwise require a different run/output location |
| GD-12 | Multi-speaker behavior has request-shape tests but no durable live acceptance | Serialization alone does not prove the provider accepts the payload | One short approved paid smoke before the release claim |

## Wave 0: execution preflight and code intelligence

No implementation card starts until repository ownership is re-confirmed on
the exact branch that will receive the change.

### Required repository routes

Every worker reads:

- `AGENTS.md`;
- `instructions/core.instructions.md`;
- `instructions/code-intelligence.instructions.md`;
- `instructions/provider-cli.instructions.md`;
- `instructions/test-quality.instructions.md`;
- `instructions/docs-governance.instructions.md`;
- `instructions/agent-kanban.instructions.md`;
- `instructions/git-release-safety.instructions.md`.

### Required Codebase evidence

Index the current checkout and answer these queries:

1. Trace `format: gemini-dialogue` from argument parsing and frontmatter
   auto-detection through validation, provider construction, payload mapping,
   generation, state, and artifacts.
2. Map every field that can change the generated dialogue: provider, model,
   top-level voice, speaker map, style prompt, prompt mode, body chunks, and
   safe tags.
3. Trace how the distributable skill is installed or referenced by supported
   agent platforms; prove whether the wheel contains it.

### Required Serena symbols

Inspect declarations, implementations, and references for:

- `validate_gemini_dialogue_file`;
- `validate_speakers`;
- `extract_speaker_voice_map`;
- `build_style_prompt`;
- `chunks_from_validation`;
- `OpenRouterTTSProvider.synthesize_chunk`;
- `_generate_step`;
- `_omnivoice_voice_identity` and the identity seam used by resume;
- `initial_state` and `load_state`;
- artifact writers that persist `speaker_voice_map`;
- parser definitions for `--format`, `--voice`, and `--speaker-voice`.

### Required ast-grep proofs

Run structural queries for:

```text
if len($MAP) > 2:
print($X)
subprocess.run($ARGS, $$$REST)
initial_state($$$ARGS)
choices=[$$$VALUES]
```

Use them to prove:

- every two-speaker cardinality check is migrated;
- no new stdout diagnostic remains in machine paths;
- state identity is passed consistently;
- provider/model/format registrations and tests remain synchronized.

If Codebase, Serena, or ast-grep is unavailable, record the exact missing tool
and block implementation rather than substituting broad recursive search.

## Wave 1: harden the existing Gemini dialogue contract

### GD-01 and GD-02: exact, distinct cast

Owner: `src/voiceover_pipeline/gemini_dialogue.py`.

Change `validate_speakers` so that:

- `len(speaker_voice_map) != 2` is an error;
- both aliases remain alphanumeric;
- both voices are from `GEMINI_TTS_VOICES`;
- the two voice names are distinct;
- error objects remain deterministic and contain no secret/provider data.

Stable error codes:

- `SPEAKER_COUNT_INVALID` for a count other than two;
- `DUPLICATE_SPEAKER_VOICE` when both aliases use the same voice;
- preserve existing `INVALID_SPEAKER_ALIAS` and `INVALID_VOICE` behavior.

No third speaker, fallback cast, or implicit duplication is introduced.

### GD-03: top-level voice is derived

Owner: `src/voiceover_pipeline/cli.py`.

For `gemini-dialogue`:

- derive the OpenRouter compatibility `voice` from the first speaker mapping;
- if the user passes `--voice`, require it to equal the derived voice;
- reject a conflict before provider construction or a paid call;
- do not add a separate top-level voice to the creative contract.

### Prompt byte safety

Owner: `src/voiceover_pipeline/gemini_dialogue.py`.

After building the shared style prompt:

- measure UTF-8 bytes;
- fail before provider construction when it exceeds the documented safe limit;
- keep the existing dialogue-body hard limit unchanged;
- do not invent provider-specific truncation.

The exact prompt limit is confirmed against the current OpenRouter/Gemini
contract during implementation. It is defined once and tested at the boundary.

### Acceptance

- One speaker fails validation.
- Three speakers fail validation.
- Two aliases with one repeated voice fail validation.
- Two aliases with two valid voices pass.
- A conflicting explicit `--voice` fails before provider construction.
- A valid cast produces the existing OpenRouter multi-speaker payload.

## Wave 2: cast-safe resume and machine-safe JSON

### GD-04: canonical dialogue identity

Owners:

- `src/voiceover_pipeline/cli.py`;
- `src/voiceover_pipeline/run_state.py` only if the existing state seam cannot
  store the identity without a schema change.

Prefer the smallest compatibility-preserving implementation. The existing
identity field may be reused if its semantics are documented as synthesis
identity rather than only OmniVoice identity.

For `gemini-dialogue`, hash a canonical UTF-8 JSON object containing:

```json
{
  "format": "gemini-dialogue",
  "provider": "openrouter-tts",
  "model": "google/gemini-3.1-flash-tts-preview",
  "speaker_voice_map": {
    "Guest": "Puck",
    "Host": "Kore"
  },
  "style_prompt_sha256": "...",
  "prompt_mode": "native"
}
```

Canonicalization rules:

- sorted object keys;
- stable speaker alias ordering;
- UTF-8 encoding;
- SHA-256 hex digest;
- no raw style prompt or transcript in public metadata;
- chunk text integrity remains covered by existing chunk hashes.

Resume behavior:

- same script, cast, model, and prompt direction: resume succeeds;
- changed speaker voice: exit code 30 before generation;
- changed model: exit code 30;
- changed style prompt/profile/vibe: exit code 30;
- old run state without dialogue identity: fail closed with a clear migration
  message; do not mix artifacts.

### GD-05: JSON stdout

Owners:

- `src/voiceover_pipeline/cli.py`;
- `src/voiceover_pipeline/providers/openrouter_tts.py`.

Requirements:

- `--json` writes exactly one JSON object to stdout;
- invalid dialogue uses the standard `{status,error,code}` envelope;
- human validation details may be nested in the object or written to stderr;
- style-prompt fallback diagnostics go to stderr;
- reject the incompatible combination `--json` + `--json-events`, or define a
  single unambiguous machine contract and test it. The minimal preferred change
  is to reject the combination before generation.

### Acceptance

- Cast changes cannot resume a paid run.
- Every dialogue failure is parseable as one JSON object.
- No fallback diagnostic appears before or after that object.
- Existing non-dialogue resume behavior remains unchanged.

## Wave 3: focused automated tests

All tests remain offline and mock the provider/network boundary.

### `tests/test_cli_validation.py`

Add deterministic cases for:

- zero/one/three speakers;
- exactly two speakers;
- duplicate voice assignment;
- invalid alias and invalid voice regression;
- conflicting top-level `--voice`;
- style-prompt byte boundary;
- body chunk byte boundary remains unchanged.

### `tests/test_new_providers.py`

Retain the request-shape regression and assert:

- exactly two `speaker_voice_configs` entries;
- speaker names match the dialogue aliases;
- voice names match the validated map;
- top-level compatibility voice equals the first validated speaker voice;
- no third speaker or raw frontmatter reaches the request.

### `tests/test_generation_stability.py`

Add dialogue resume tests:

- same cast resumes and skips completed chunks;
- changing either voice is rejected;
- changing model is rejected;
- changing profile/vibe/style prompt is rejected;
- old state without identity fails closed.

### `tests/test_cli_json_contract.py`

Add agent-facing machine cases:

- valid `validate --format gemini-dialogue --agent --json`;
- invalid dialogue under `generate --json` returns one error object;
- provider fallback diagnostics stay off stdout;
- `--json` + `--json-events` is rejected deterministically.

### Mocked end-to-end generation

Add one test that exercises:

1. Frontmatter auto-detection.
2. Dialogue validation.
3. OpenRouter provider construction.
4. Multi-speaker payload serialization.
5. Fake audio response conversion.
6. `chunks.json` with `script_format` and `speaker_voice_map`.
7. `run_state.json` with dialogue identity.
8. Final one-object JSON response.

This is the release's primary offline proof. No GPU, API key, `.env`, network,
or paid request is allowed.

## Wave 4: agent skill reconciliation

The skill source already exists under `docs/skills/voiceover-pipeline/`. Do not
create a second skill.

### `docs/skills/voiceover-pipeline/SKILL.md`

Make these minimal changes:

- Add an explicit **F: Two-speaker podcast** branch.
- Route "podcast", "dialogue", "two hosts", and "question and answer" to
  `gemini-dialogue` unless the user explicitly requires a local-only result.
- Change the boundary from "does not write scripts" to:
  "does not invent unrelated creative content; when the user requests a
  complete voiceover or podcast artifact, the agent may author the structured
  script needed to produce it."
- Keep provider selection explicit before a paid call.
- State that local OmniVoice is single-speaker per run in `0.6.0`.
- Update provider count and compatibility version.

### `docs/skills/voiceover-pipeline/docs/04-input-format.md`

- Require exactly two distinct speaker voices.
- Keep the current frontmatter format and safe tags.
- Add a compact Q/A example using `Host` and `Guest`.
- Document that `******` creates semantic request chunks, not separate casts.

### `docs/skills/voiceover-pipeline/docs/06-commands-and-flags.md`

- Add `omnivoice-local` to provider discovery.
- Document `auto`, bank `preset`, `clone`, `design`, and `--voice-bank`.
- Document `gemini-dialogue` and repeatable `--speaker-voice` overrides.
- Keep local and paid-provider flags clearly separated.

### `docs/skills/voiceover-pipeline/docs/08-workflows.md`

Add one canonical agent workflow:

1. Author `podcast.md` from the user's requested topic and cast.
2. Run `validate --agent --json`.
3. Run `doctor --provider openrouter-tts --json` without reading `.env`.
4. Ask for paid generation approval if it has not already been granted.
5. Run `generate --json --resume`.
6. Read `manifest.json`, `run_state.json`, and `generation.log`.
7. If timings are needed, prefer `generate --with-timings` in the same safe
   generation flow.

Remove the destructive recommendation to run standalone timings with the same
output directory/run ID and `--overwrite`. If standalone timings are necessary,
use a distinct output directory and run ID.

Also add concise OmniVoice workflows for:

- default voice-bank preset;
- named bank profile;
- ad-hoc clone;
- design instruction;
- auto voice.

### `docs/skills/voiceover-pipeline/docs/14-local-audio-cpp-models.md`

Replace the obsolete fixed-female-only OmniVoice description with the shipped
mode matrix and the current native Windows acceptance boundary.

### Example

Add:

```text
docs/skills/voiceover-pipeline/examples/gemini-dialogue-podcast.md
```

The example is short, uses placeholders, contains no secret, and is directly
validatable by the CLI.

### Skill availability gate

The repository contains the skill source but no repository-owned installer was
found during planning, and the current agent session did not expose
`voiceover-pipeline` as an available skill.

For `0.6.0`:

- use the agent platform's existing skill installation mechanism;
- document the exact installation command/path after verifying it against the
  current platform documentation;
- do not add a custom installer unless the platform has no supported mechanism;
- run trigger smoke checks after installation:

```text
should trigger: "сделай подкаст с двумя ведущими"
should trigger: "озвучь диалог мужчины и женщины"
should trigger: "create a Gemini multi-speaker podcast"
should not trigger: "напиши сценарий, но не озвучивай"
```

Acceptance requires the installed skill to appear in the agent's available
skill inventory and route the first three prompts to the canonical workflow.

## Wave 5: machine contract and primary documentation

### `docs/agent-cli-contract.md`

Add a machine-facing Gemini dialogue section covering:

- frontmatter schema;
- exactly two distinct speakers;
- model/provider restriction;
- voice allowlist;
- safe tags and UTF-8 byte limits;
- derived top-level compatibility voice;
- JSON and exit-code behavior;
- dialogue resume identity;
- artifact fields and current turn-level limitations.

Do not duplicate prompting advice already owned by the skill tree.

### `docs/openrouter-tts-models.md`

Document the production multi-speaker request shape and distinguish:

- offline/mocked contract proof;
- live provider acceptance;
- volatile model availability and pricing.

### `README.md`

Add one short agent-first podcast example and correct the provider/capability
summary. Avoid copying the full contract.

### `CHANGELOG.md`

Prepare the `0.6.0` entry with:

- OmniVoice voice bank and four local modes;
- native Windows acceptance;
- hardened Gemini dialogue workflow;
- agent skill reconciliation;
- resume/JSON safety fixes;
- known limitation: local multi-speaker dialogue is not first-class.

### `docs/README.md`

This file currently has a pre-existing user modification. Do not overwrite or
stage it blindly. Reconcile the user's change before adding or updating an index
link. If no safe merge is available, leave the plan reachable through
`docs/plans/` and record the deferred index update.

## Wave 6: release version and package verification

Target version: `0.6.0` because the current branch includes new public
OmniVoice capabilities and a hardened agent dialogue workflow, not only a patch.

Update consistently:

- `pyproject.toml` project version;
- `src/voiceover_pipeline/__init__.py` `__version__`;
- `uv.lock` root package version via the normal lock command;
- skill compatibility line;
- changelog heading and release references.

Do not hand-edit unrelated lock entries.

### Required verification

Run in this order:

```powershell
uv run pytest tests/test_cli_validation.py `
  tests/test_new_providers.py `
  tests/test_generation_stability.py `
  tests/test_cli_json_contract.py -q

uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy --no-incremental
uv run pytest -q
uv build
```

Then inspect locally:

- wheel and sdist names report `0.6.0`;
- package metadata reports `0.6.0`;
- installed CLI reports the same version;
- wheel contains the expected Python modules;
- the distributable skill path and installation instructions point to the
  immutable `0.6.0` tag/revision;
- no `.env`, audio, model, cache, local path, or secret is present.

Existing whole-tree failures outside the changed scope are recorded exactly.
They are not hidden by changing configuration or weakening tests.

## Wave 7: live Gemini acceptance

This wave is a paid network operation and starts only after explicit approval.

Use one short Russian script:

- exactly two speakers;
- two distinct Gemini voices;
- two to four turns;
- under the safe request byte limit;
- no private or user-provided content;
- fresh output directory and run ID.

Capture only safe evidence:

- provider/model IDs;
- speaker aliases and public voice names;
- request/response success booleans;
- output duration and format;
- artifact hashes;
- JSON contract and cleanup booleans;
- cost metadata returned by the provider, if available.

Do not commit raw audio, prompt text, provider response bodies, API keys, local
paths, or `.env` data.

Acceptance:

- both voices are audible and correctly assigned;
- no role swapping;
- MP3 and manifests exist;
- `--json` is one object;
- resume with unchanged cast succeeds without regenerating completed chunks;
- resume after cast modification is rejected before a paid request;
- no secrets appear in logs or artifacts.

If the current OpenRouter model or payload is rejected, do not add speculative
fallback schemas. Mark live acceptance blocked, preserve the mocked contract,
and investigate the current provider documentation separately.

## Commit sequence

Keep commits narrow and reviewable:

1. `fix: harden Gemini dialogue cast and resume identity`
2. `test: cover agent Gemini dialogue generation contract`
3. `docs: align voiceover skill with dialogue and OmniVoice`
4. `chore: prepare voiceover-pipeline 0.6.0`
5. `docs: record Gemini dialogue live acceptance` (only after approved live
   smoke)

Before every commit:

```powershell
git status --short
git diff --check
git diff --cached
git log --oneline -10
```

Stage only intended files. Preserve the existing user modification in
`docs/README.md` and the untracked `.serena/` and `in/` paths.

Push, tag, package publication, and GitHub release are separate operations. A
successful implementation push does not authorize a tag or registry publish.

## Stop conditions

Stop and report rather than expanding scope when:

- required code-intelligence tools are unavailable;
- the exact OpenRouter multi-speaker schema cannot be confirmed;
- dialogue resume cannot be made cast-safe without a state migration larger
  than this release;
- machine JSON cannot remain one-object compatible;
- skill installation requires inventing an unsupported agent-platform
  mechanism;
- local or paid live acceptance would require reading `.env` or exposing a key;
- current provider behavior requires a direct Google SDK rewrite;
- unrelated working-tree changes conflict with a required file.

## Definition of done

`0.6.0` is ready to tag only when all statements are true:

- A valid two-speaker script passes validation.
- Any count other than two fails before provider construction.
- Duplicate voice assignment fails before provider construction.
- A conflicting top-level voice fails before provider construction.
- The OpenRouter request contains exactly two speaker configs.
- Dialogue identity covers model, cast, style direction, and prompt mode.
- Resume rejects cast/model/style changes before a paid call.
- `--json` produces one object for success and failure.
- The mocked end-to-end dialogue generation test passes.
- Existing non-dialogue providers and OmniVoice modes remain green.
- The main skill routes podcast requests to `gemini-dialogue`.
- Agents may author the structured script when the user requests the complete
  artifact.
- Skill guidance documents current OmniVoice bank/clone/design/auto modes.
- Unsafe same-run standalone timings guidance is removed.
- The skill is installable through the supported platform mechanism and passes
  trigger smoke checks.
- Project, runtime, lock, skill, and changelog versions agree on `0.6.0`.
- Focused tests, Ruff, mypy, full pytest, and local package build are recorded.
- One explicitly approved live two-speaker smoke passes, or the release claim is
  explicitly limited to offline/mock acceptance.
- No tag, package publication, or release is performed without its own approval.

## Deferred backlog after 0.6.0

These items are useful but deliberately excluded from the release critical
path:

- generic provider-neutral dialogue model;
- local OmniVoice/Qwen alternating-speaker renderer;
- per-turn audio and timing artifacts;
- direct Google Gemini provider;
- voice-bank creation/admission CLI;
- automatic topic research and script generation inside VOP;
- append-to-existing-run standalone timing workflow;
- richer cast metadata in the top-level production manifest.

## Execution ledger

| Wave | Status | Evidence |
|---|---|---|
| 0. Code intelligence | pending | Codebase, Serena, ast-grep proofs |
| 1. Dialogue validation | pending | focused validation tests |
| 2. Resume and JSON | pending | stability and JSON tests |
| 3. Automated coverage | pending | mocked end-to-end generation |
| 4. Agent skill | pending | installed skill + trigger smoke |
| 5. Contracts/docs | pending | link/frontmatter/diff checks |
| 6. Version/package | pending | `uv build` + artifact inspection |
| 7. Live acceptance | pending approval | safe paid smoke record |
