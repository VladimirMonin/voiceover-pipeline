---
name: Agent Kanban workflow
description: Task ownership and status rules for agents working through the voiceover-pipeline board.
applyTo: "**"
---

# Agent Kanban workflow

- Use the Kanban board named `voiceover-pipeline` for repository work when board tooling is available.
- Before implementation, find the relevant card, confirm scope and acceptance criteria, and move/mark only that card as in progress. Do not create duplicates when an existing card covers the task.
- Keep card notes concise: decisions, blockers, changed files, and verification evidence. Never put secrets, `.env` contents, tokens, or sensitive logs on the board.
- Do not mark work complete until acceptance criteria are met and verification has actually run. Record partial verification or blockers explicitly.
- Do not alter unrelated cards, priorities, ownership, or board structure without explicit instruction.
- If board tooling is unavailable, continue only when the task is otherwise unambiguous and report that board state could not be updated.
