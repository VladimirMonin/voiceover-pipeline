---
name: Core repository contract
description: Baseline scope, source-of-truth, and change-safety rules for every task.
applyTo: "**"
---

# Core repository contract

- Read `AGENTS.md` and every instruction whose `applyTo` matches the files being handled.
- Inspect the relevant implementation, public docs, and tests before changing behavior; do not infer interfaces from stale prose.
- Make the smallest coherent change. Do not reformat, regenerate, delete, or restore unrelated files.
- Treat a dirty working tree as user-owned state. Record `git status --short` before and after work and preserve pre-existing modifications and untracked paths.
- Never read, print, source, parse, copy, or expose `.env`, credential files, API keys, tokens, or secret-bearing command output. Check configuration only through redacted application interfaces such as `voiceover doctor --json` when the task explicitly allows execution.
- Do not perform paid, cloud, live-provider, upload, publish, or other network operations without explicit user approval.
- Keep generated audio, run output, caches, and local environment files out of instruction-only changes.
- Report exactly what was changed, what was verified, and what remains unverified.
