# AGENTS.md — voiceover-pipeline

## Owner's agent workflow

- Act as a thin orchestrator: assign narrow implementation tasks; the parent owns Git scope and final verification. Keep one writer per worktree and delegate write or Git ownership explicitly.
- Use independent review for substantial stages or material risks, not every small correction. Consult the expert when genuinely blocked.
- Run offline tests and already-authorized Git steps without asking again. After a confirmed stage, make a scoped commit and push when the owner has explicitly authorized that push; do not infer approval for live/paid/cloud calls, tags, releases, or secret handling.

| Pi role | Agent | Default provider/model (owner-editable here) |
|---|---|---|
| Main coder | `worker` | `ollama-cloud/deepseek-v4.1-flash` |
| Stage reviewer | `reviewer` | `openai-codex/gpt-6-sol` |
| Blocker expert | `oracle` | `openai-codex/gpt-6-astra` |

Use `ollama-cloud/deepseek-v4.1-flash` as the primary `worker` route. If it is unavailable or quota-limited, `polza/deepseek/deepseek-v4.1-flash` is the owner-approved backup for the same Pi subagent role; report the switch rather than substituting silently. Check current agent/model availability before delegation. This model routing does not authorize live/paid calls by the application.

Repository instructions are split into small, scoped files. Read this router first, then load every matching file from `instructions/` before acting. Repository research uses Codebase → Serena → ast-grep. When accepted behavior changes a fundamental durable rule, updating the owning instruction is mandatory; follow `instructions/instruction-authoring.instructions.md`.

## Instructions

Every file in `instructions/` has exactly one link below. Listing is not loading: load every `Always` route on each task, and load a conditional route only when its condition matches the files and work you are handling.

- Load always — [`instructions/core.instructions.md`](instructions/core.instructions.md): baseline scope, source-of-truth, and change-safety rules.
- Load always — [`instructions/code-intelligence.instructions.md`](instructions/code-intelligence.instructions.md): Codebase → Serena → ast-grep research order.
- Load always — [`instructions/agent-kanban.instructions.md`](instructions/agent-kanban.instructions.md): agent task tracking.
- Load always — [`instructions/git-release-safety.instructions.md`](instructions/git-release-safety.instructions.md): Git, versioning, packaging, and releases.
- Load when creating or editing `AGENTS.md` or `instructions/**` — [`instructions/instruction-authoring.instructions.md`](instructions/instruction-authoring.instructions.md): instruction ownership, routing, and maintenance.
- Load for provider, model, prompt, timing, or CLI behavior — [`instructions/provider-cli.instructions.md`](instructions/provider-cli.instructions.md).
- Load for tests or behavior changes — [`instructions/test-quality.instructions.md`](instructions/test-quality.instructions.md).
- Load for documentation — [`instructions/docs-governance.instructions.md`](instructions/docs-governance.instructions.md).

## Source-of-truth map

- Project overview and development setup: [`README.md`](README.md), [`pyproject.toml`](pyproject.toml)
- Machine-facing CLI behavior: [`docs/agent-cli-contract.md`](docs/agent-cli-contract.md)
- Documentation index: [`docs/README.md`](docs/README.md)
- Agent development workflow: [`doc/agent-workflow.md`](doc/agent-workflow.md)
- User-facing skill: [`docs/skills/voiceover-pipeline/SKILL.md`](docs/skills/voiceover-pipeline/SKILL.md)
- Implementation: `src/voiceover_pipeline/`; tests: `tests/`

## Non-negotiable safety

Never read, print, source, parse, copy, or expose `.env` or secret values. Never commit secrets. Publishing, tagging, pushing, and any live/cloud or other network operation require explicit user approval for that operation. Preserve unrelated and pre-existing working-tree changes.
