---
name: Git and release safety
description: Safe working-tree, commit, version, tag, push, build, and package publication contract.
applyTo: "{**,.gitignore,pyproject.toml,uv.lock,CHANGELOG.md,scripts/**,docs/**,doc/**,instructions/**}"
---

# Git and release safety

- Begin and end with `git status --short`; inspect only scoped diffs. Never discard, stage, rewrite, delete, or include unrelated or pre-existing dirty files.
- Do not use destructive Git commands (`reset --hard`, `clean`, forced checkout/restore, force push) unless the user explicitly requests the exact destructive action after reviewing impact.
- A request to edit files is not approval to commit. A request to commit is not approval to tag, push, publish, upload, create a release, or contact a registry. Obtain explicit approval for each network/release action.
- Never read or source `.env`; never place credentials on a command line, in logs, config, commits, tags, artifacts, or board notes. Use the publisher/CI credential mechanism approved by the user without exposing values.
- Before an approved release, require a clean, understood release scope; verify version and changelog consistency, tests, build artifacts, package metadata/content, and target registry. Build locally before publication where permitted.
- Create no tag until the exact version/commit is confirmed. Use no push or package publication dry-run against a remote unless explicitly approved; dry-run can still be a network operation.
- After any approved commit/tag/publish step, report immutable identifiers and actual command outcomes. Never claim a release succeeded without registry/remote confirmation.
- For instruction-only work: no commit, push, tag, build, lockfile update, live/cloud call, or publication; use structural checks and `git diff --check` only.
