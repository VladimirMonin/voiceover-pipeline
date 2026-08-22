# Agent-first fix plan: two-voice dialogue for OpenRouter and OmniVoice

> **Status:** approved by user; implementation starts after this plan is committed.
> Executed by sub-agents in waves; each wave lands as a separate commit.
> No paid provider call, release tag, or publication happens without separate
> approval at its gate. `0.6.0` is **not** tagged or published until two human
> PASS acceptances (OpenRouter + OmniVoice) are recorded.

## Root cause (confirmed)

- OpenRouter `/api/v1/audio/speech` documents a single top-level `voice`
  (https://openrouter.ai/docs/guides/overview/multimodal/tts). The field
  `multi_speaker_voice_config` is **not** part of that endpoint.
- The previous implementation sent a hybrid payload: top-level `voice=Kore`
  plus an unverified snake_case `multi_speaker_voice_config`. OpenRouter
  accepted the request, ignored the undocumented field, and synthesized the
  whole input with `Kore` — the user heard one female voice.
- `run_state.json` recorded `"voice": "Kore"` for every chunk; that was the
  blocking signal, and it was missed.
- Mocked tests only asserted the local JSON shape and always returned audio;
  they could not prove the provider applied two voices.
- Acceptance inferred role correctness from request metadata while the
  audibility line remained OPEN. Audible cast assignment actually FAILED.
- Real native multi-speaker exists only on the direct Gemini API via
  `generation_config.speech_config` with two `{speaker, voice}` entries
  (https://ai.google.dev/gemini-api/docs/speech-generation). It is not
  supported through OpenRouter.

## Current OmniVoice reality

- One global profile is bound for the whole run; the provider is created once;
  OmniVoice fragments are deliberately merged into one native session;
  `--speaker-voice` is rejected for OmniVoice; resume stores one scalar voice
  identity. Two voices in one run are **not** supported today.
- The approved voice bank (`C:\audio-cpp-work\voice-bank\approved\`) already
  contains `omni-female-neutral-01`, `omni-female-deep-01`,
  `omni-male-neutral-01`, `omni-male-deep-01`.
- The native runtime can call different profiles sequentially; no C++ change
  is needed. What is needed is a Python dialogue router: one turn = one
  native call with the speaker's profile.

## Design decisions

1. **Canonical format** is `dialogue` (provider-independent). `gemini-dialogue`
   stays as a compatibility alias (spelling is already fixed in the v0.5.1 tag).
   State/manifests always write `format: dialogue`.
2. **Internal execution strategy** `turn-by-turn-v1`, never exposed to users.
3. **One turn = one provider request with exactly one voice**:
   - OpenRouter: documented payload only — `model`, `input`, `voice`,
     `response_format`; `multi_speaker_voice_config` removed entirely.
   - OmniVoice: one native call per turn with the speaker's bank profile.
   - The `Alias:` label is never part of the synthesized text.
4. **Deterministic pauses**: trim speech first, then insert local PCM silence —
   250 ms between turns, 600 ms after `******` section separator, 0 ms after
   the final turn. Stitching follows the dialogue plan order, not
   lexicographic `chunk_*.mp3`.
5. **New resume identity** `synthesis_identity` (execution strategy,
   provider/model, alias → voice/profile/fingerprint, ordered turns with text
   hashes and pauses, style/prompt hashes, OmniVoice seed/steps/guidance,
   trim/pause policy). Any cast/voice/fingerprint/text/order/pause change →
   exit 30 before any provider call. Old dialogue state without the new
   identity fails closed. Orphan MP3s are never resumed without trusted state.
6. **Per-turn artifacts**: `turn_index`, `speaker`, `voice`,
   `voice_fingerprint`, `speech_duration_ms`, `pause_after_ms`, `audio_sha256`,
   `start_ms`, `end_ms`. No reference paths/transcripts or voice-bank private
   content in manifests/logs.
7. **Direct Gemini provider** (`google-gemini-tts`, official Interactions API
   with `speech_config`) is a separate, non-blocking track — do not attempt to
   smuggle that contract through OpenRouter again.
8. Voice gender is never inferred from names: control clips are listened to by
   the user before cast assignment.

## Wave plan (subagents)

### Wave 0 — capability spikes (before implementation)

- **OmniVoice A/B/A spike (local, free):** three native calls, same control
  text and same seed for A1/A2; profiles
  `omni-female-neutral-01` / `omni-male-deep-01` / female again; verify
  distinct catalog fingerprints and user confirms female/male/female. If
  single calls work, orchestration is technically feasible.
- **Docs fail (commit 1):** mark live acceptance report FAILED/BLOCKED
  (transport and audio generation passed; audible cast assignment failed),
  remove from contracts every claim that OpenRouter applies native
  multi-speaker, and block the current hybrid OpenRouter payload.
- **OpenRouter paid spike** is deferred until explicit user approval and
  listening (control A/B/A with two candidate voices).

### Wave 1 — Foundation

- `DialogueTurn` model + `ScriptChunk` extension (speaker/voice/pause,
  no `Alias:` in text).
- Provider-independent dialogue planner: canonical `dialogue` format,
  `gemini-dialogue` alias → identical plan; section separator semantics;
  per-turn voice routing table.
- Exit-30 conflict check for `--voice` vs script speakers (existing guard
  preserved).

### Wave 2 — Per-turn providers

- **OpenRouter:** one bound provider per voice; per-turn top-level `voice`;
  strip `multi_speaker_voice_config` and all undocumented fields.
- **OmniVoice:** catalog/runtime admitted once per run; two lightweight
  profile bindings sharing the admitted runtime; per-turn native call with the
  turn profile reference; no `merge_omnivoice_session_fragments` for dialogue;
  sequential execution; distinct fingerprints required for two speakers
  (same fingerprint with different IDs rejected).

### Wave 3 — Integration

- CLI wiring: dialogue plan → turn synthesis loop; deterministic pause/stitch
  in plan order; per-turn artifacts; resume identity + fail-closed rules;
  JSON remains a single object.

### Wave 4 — Tests + docs

- Mandatory matrix (see below), then docs: skill tree, `docs/agent-cli-contract.md`,
  `docs/openrouter-tts-models.md` multi-speaker claim removal, README, plan log.

### Wave 5 — Live acceptance (approval gates)

- **OpenRouter** (paid, requires separate approval): control A/B/A + integrated
  dialogue A/B/A/B; captured request evidence (one documented voice per turn);
  human listening mandatory; resume issues no new paid requests; cast change
  rejected before network; acceptance closes only when audibility is PASS.
- **OmniVoice** (local): control A/B/A, integrated female/male/female,
  request receipts per turn, human listen, back-transcription, deterministic
  repeat, resume, cast-change rejection, injected failure on turn 2 proves
  turn 3 never runs, zero leftover processes, no reference-data leaks.

## Auto-test matrix (Wave 4, offline only)

- `dialogue` and `gemini-dialogue` alias produce the same plan.
- One turn = one provider request.
- Speaker label never appears in synthesized text.
- OpenRouter A/B/A produces `voice=[A,B,A]`.
- No `multi_speaker_voice_config` in the OpenRouter payload.
- OmniVoice A/B/A selects the correct profile IDs and fingerprints.
- Catalog/runtime admission for OmniVoice happens once per run.
- Different IDs with the same fingerprint are rejected.
- OmniVoice never merges turns of different speakers.
- Pauses, offsets, final zero-pause are deterministic.
- Resume rejects cast/text/order/fingerprint/pause-policy changes.
- Orphan dialogue audio is not resumed.
- >99 turns stitch in numerical order.
- `--json` still emits a single JSON object.
- No reference material in manifests/logs.

## Commit sequence

1. `docs: mark Gemini dialogue live acceptance failed`
2. `fix: block unsupported OpenRouter multispeaker payload`
3. `feat: add provider-neutral dialogue turn plan`
4. `feat: route OpenRouter dialogue per speaker turn`
5. `feat: route OmniVoice dialogue per bank profile`
6. `fix: harden dialogue resume pauses and artifacts`
7. `test: prove two-speaker dialogue routing`
8. `docs: align dialogue skill and machine contract`
9. `docs: record OpenRouter and OmniVoice audible acceptance`

Version stays `0.6.0` (never tagged or published). The tag is allowed only
after both human PASS results are recorded.

## Safety gates

- No `.env` reading; no secrets in commits, manifests, logs, or board notes.
- Preserve user-owned untracked paths (`.serena/`, `in/*`).
- Ruff lint, Ruff format check, mypy `--no-incremental` on every changed file.
- Commit after each wave; push only when the user asks.
