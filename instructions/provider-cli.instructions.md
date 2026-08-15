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
- Do not promise current model availability, price, latency, or quality from historical examples; label snapshots and verify current facts only with approved network access.
