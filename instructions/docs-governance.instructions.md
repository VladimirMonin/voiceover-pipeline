---
name: Documentation governance
description: Routing, ownership, consistency, and maintenance rules for repository documentation.
applyTo: "{README.md,AGENTS.md,doc/**/*.md,docs/**/*.md,instructions/**/*.md}"
---

# Documentation governance

- Keep `AGENTS.md` a compact router. Put durable rules in atomic `instructions/*.instructions.md` files with valid `name`, `description`, and `applyTo` frontmatter.
- Use `docs/README.md` as the documentation index and `doc/agent-workflow.md` as the contributor-agent workflow. Link new durable documents from the appropriate index.
- `docs/agent-cli-contract.md` owns machine-facing CLI semantics. The skill tree under `docs/skills/voiceover-pipeline/` owns distributable end-user agent guidance; avoid silently duplicating contracts.
- Verify commands, flags, provider IDs, paths, and schema names against source interfaces and tests before documenting them.
- Avoid volatile facts in permanent guidance: current test counts, unqualified prices/availability, one-off run results, and release history belong in reports or explicit changelogs.
- Use relative repository links and ensure every routed target exists. Keep one primary topic per atomic instruction.
- Documentation examples must use placeholders, never secret values, and must not teach reading/sourcing `.env`.
- For docs-only changes, verify frontmatter, local links, scoped diff, and whitespace with `git diff --check`.
