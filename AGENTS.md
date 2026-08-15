# AGENTS.md — voiceover-pipeline

Repository instructions are split into small, scoped files. Read this router first, then load every matching file from `instructions/` before acting.

## Required routes

- Always: [`instructions/core.instructions.md`](instructions/core.instructions.md)
- Provider, model, prompt, timing, or CLI behavior: [`instructions/provider-cli.instructions.md`](instructions/provider-cli.instructions.md)
- Tests or behavior changes: [`instructions/test-quality.instructions.md`](instructions/test-quality.instructions.md)
- Documentation: [`instructions/docs-governance.instructions.md`](instructions/docs-governance.instructions.md)
- Agent task tracking: [`instructions/agent-kanban.instructions.md`](instructions/agent-kanban.instructions.md)
- Git, versioning, packaging, or releases: [`instructions/git-release-safety.instructions.md`](instructions/git-release-safety.instructions.md)

## Source-of-truth map

- Project overview and development setup: [`README.md`](README.md), [`pyproject.toml`](pyproject.toml)
- Machine-facing CLI behavior: [`docs/agent-cli-contract.md`](docs/agent-cli-contract.md)
- Documentation index: [`docs/README.md`](docs/README.md)
- Agent development workflow: [`doc/agent-workflow.md`](doc/agent-workflow.md)
- User-facing skill: [`docs/skills/voiceover-pipeline/SKILL.md`](docs/skills/voiceover-pipeline/SKILL.md)
- Implementation: `src/voiceover_pipeline/`; tests: `tests/`

## Non-negotiable safety

Never read, print, source, parse, copy, or expose `.env` or secret values. Never commit secrets. Publishing, tagging, pushing, and any live/cloud or other network operation require explicit user approval for that operation. Preserve unrelated and pre-existing working-tree changes.
