# Agent development workflow

This document describes how repository agents plan, execute, and verify changes. It does not replace the scoped contracts in [`../instructions/`](../instructions/).

## 1. Discover and protect state

1. Read [`../AGENTS.md`](../AGENTS.md), [`../instructions/code-intelligence.instructions.md`](../instructions/code-intelligence.instructions.md), and all other matching atomic instructions.
2. For repository research, index/map with Codebase, narrow exact symbols with Serena, and prove structural patterns with ast-grep before direct targeted reads.
3. Inspect `git status --short` and treat every existing modification/untracked path as user-owned.
4. Read only the implementation owners narrowed by code intelligence, plus relevant public docs and tests. Never inspect `.env` or reveal credentials.
5. On the `voiceover-pipeline` Kanban board, locate the relevant card and confirm that its body explicitly requires Codebase, Serena, ast-grep, exact instruction paths, and measurable evidence before dispatch.

## 2. Define the change boundary

- List files that may change and files that must remain untouched.
- Separate behavior, tests, documentation, and release work.
- Get explicit approval before live/cloud, paid, network, destructive, publish, tag, or push operations.
- Do not infer permission for a later release step from permission for an earlier one.

## 3. Implement safely

- Make the smallest coherent edits.
- Preserve CLI JSON/stdout/stderr and semantic exit-code contracts unless a breaking change is intentional and approved.
- Use mocked offline tests for providers; never use real credentials.
- Keep durable rules atomic and route them through `AGENTS.md`.

## 4. Verify

For code changes, run focused offline tests, then broader tests as scope permits. For documentation/instruction-only changes:

1. Check YAML frontmatter fields (`name`, `description`, `applyTo`).
2. Check routed local links and target existence.
3. Review only the scoped diff.
4. Run `git diff --check`.
5. Re-run `git status --short` and confirm pre-existing dirty paths are unchanged.

## 5. Close and report

Update only the relevant Kanban card with files changed, verification evidence, and blockers. Mark it complete only when acceptance criteria are met. Report actual outcomes and clearly identify any checks not run. Do not commit, tag, push, or publish unless separately and explicitly requested.
