---
name: Provider and CLI contract
description: Rules for provider implementations, CLI behavior, machine output, and paid generation safety.
applyTo: "{src/voiceover_pipeline/**/*.py,tests/**/*.py,docs/agent-cli-contract.md,docs/**/*provider*.md,docs/skills/voiceover-pipeline/**/*.md}"
---

# Provider and CLI contract

- Treat `docs/agent-cli-contract.md` and executable tests as the public compatibility contract; reconcile either with implementation when behavior changes.
- Preserve machine mode: `--json` writes one parseable JSON object to stdout, diagnostics go to stderr, and semantic exit codes remain stable unless an intentional breaking change is approved and documented.
- Validate provider/model/voice combinations and destructive output paths before requests or filesystem deletion.
- Provider additions or changes must cover registration/listing, defaults, required-key detection, request/response mapping, failures, cost metadata where available, CLI docs, and mocked tests.
- Tests must mock provider/network calls. Never use real keys, paid requests, live endpoints, or user `.env` data.
- Prefer resumable generation and existing-output safeguards. Do not use overwrite for paid generation unless the user explicitly chooses that loss/cost risk.
- For paid TTS, persist an attempt marker before submit and preserve it as evidence until the outcome is confirmed and the part is saved. Never automatically resubmit an unconfirmed attempt via outer retry, provider fallback, resume, or overwrite; keep local-provider retry separate. For a paid Polza TTS ElevenLabs `/media` submit, a stored bounded accepted task id may be finished by an explicit `--resume` with GET poll/download only — never a second POST — while provider, model, voice, and script identity still match and every earlier chunk file is on disk; `--overwrite` stays blocked for any marker.
- Preserve accepted paid audio in a run-local raw file with a bounded receipt before FFmpeg. A matching on-disk receipt may resume local conversion without another provider request; a missing or corrupt receipt never permits a fresh paid submit (a separately valid Polza Media task id may still recover by GET only). Retain paid raw after success rather than treating it as disposable cache.
- Once a run has committed native ownership or carries native local evidence (ownership descriptor, `native_history` run state, or a raw receipt), never fall back to the legacy JSON writer; serialize ownership selection and mutation of both native and legacy writers under the shared run-root lock, and write JSON only as a DB-derived compatibility export for native runs.
- Do not promise current model availability, price, latency, or quality from historical examples; label snapshots and verify current facts only with approved network access.
