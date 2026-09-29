---
name: Instruction authoring and maintenance
description: Use when creating, renaming, splitting, retiring, or updating files in instructions/ — durable-rule threshold, frontmatter triggers, AGENTS.md routing, and validation for atomic repository instructions.
applyTo: "{AGENTS.md,instructions/**}"
---

# Instruction authoring and maintenance

## Fundamentals-only threshold

- An instruction holds only durable rules that change how future tasks are performed: contracts, safety constraints, workflow, routing, and ownership. One small topic per file.
- Do not add one-off task results, run logs, transient status, current counts, prices, or release history. If the rule would be stale after the next task, it is not fundamental and must not be added.
- Never place secrets, credentials, or `.env` reading/sourcing guidance in instructions.
- Keep files lean. Prefer the shortest wording that states the rule, and split a file that grows a second independent topic. The hard maximum is 600 lines per file.
- Do not add speculative or transient instructions for work that does not exist yet.

## Required frontmatter

- Every instruction starts with YAML frontmatter containing `name`, `description`, and `applyTo`.
- `description` is a trigger, not a summary: name the subsystem and the tasks, files, or terms that make the instruction relevant.
- `applyTo` must match exactly when the rule is needed; use quoted brace/glob forms and do not use `"**"` for narrow topics. Reserve `"**"` for repository-wide contracts such as core, code intelligence, Kanban, and Git/release safety.
- Every instruction must be reachable from an `AGENTS.md` route with a clear topic description, and `AGENTS.md` must list every file in `instructions/`.

## Add, update, split, retire

1. Update the instruction that already owns the topic before creating a new file.
2. Create a new instruction only for a durable topic no existing instruction covers, then add its `AGENTS.md` route in the same change.
3. Split a file when it carries two independent responsibilities, and update `AGENTS.md` and any inbound routes.
4. Retire a topic by deleting the file and removing its `AGENTS.md` route in the same change.
5. When accepted repository behavior changes a fundamental rule, refresh the owning instruction in the same change instead of leaving stale guidance.
6. Do not duplicate guidance another instruction or `docs/` document owns; link the owner. Documentation and link policy are owned by `instructions/docs-governance.instructions.md`.

## Validate before reporting

- Check that `name`, `description`, and `applyTo` are present, that every `AGENTS.md` route resolves to an existing file, and that instruction links point at real targets.
- Review only the scoped diff and run `git diff --check`.
- `git diff --check` does not see a new untracked file, so confirm its whitespace explicitly, for example `git diff --no-index --check /dev/null <file>`. A nonzero exit only means the file differs; the diagnostics must be empty.
