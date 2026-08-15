---
name: Agent Kanban workflow
description: Task ownership and status rules for agents working through the voiceover-pipeline board.
applyTo: "**"
---

# Agent Kanban workflow

- Use the Kanban board named `voiceover-pipeline` for repository work when board tooling is available.
- Before implementation, find the relevant card, confirm scope and acceptance criteria, and move/mark only that card as in progress. Do not create duplicates when an existing card covers the task.
- Every research or code-changing card must require the worker to read `AGENTS.md`, `instructions/core.instructions.md`, `instructions/code-intelligence.instructions.md`, and all subsystem instructions matching its files.
- Card bodies must name the required Codebase index/query, Serena symbol/reference checks, and ast-grep structural query. Read back the body before dispatch; correct any card that omits one of these routes.
- Worker handoffs must cite actual Codebase, Serena, and ast-grep evidence rather than claiming that tools were used.
- Keep card notes concise: decisions, blockers, changed files, and verification evidence. Never put secrets, `.env` contents, tokens, or sensitive logs on the board.
- Do not mark work complete until acceptance criteria are met and verification has actually run. Record partial verification or blockers explicitly.
- Do not alter unrelated cards, priorities, ownership, or board structure without explicit instruction.
- If board tooling is unavailable, continue only when the task is otherwise unambiguous and report that board state could not be updated.
