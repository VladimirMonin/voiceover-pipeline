---
name: Tool-first code intelligence
description: Mandatory Codebase, Serena, and ast-grep routing for repository research, architecture, implementation planning, and Kanban cards.
applyTo: "**"
---

# Tool-first code intelligence

- Every agent, coordinator, subagent, and Kanban worker must read `AGENTS.md` and every matching file in `instructions/` before researching or changing this repository.
- Do not begin code research with broad direct file reading, recursive grep, or sequential repository browsing. Direct reads are allowed only after code-intelligence tools have narrowed the exact owners, symbols, or structural matches.
- For architecture, provider integration, model adapters, prompting, timing, CLI wiring, or cross-layer impact work, use all three routes:
  1. **Codebase** — index the current repository and map cross-file/cross-layer owners and dependencies.
  2. **Serena** — inspect exact symbols, declarations, implementations, and references after owners are known.
  3. **ast-grep** — prove structural patterns, registration shapes, duplicated forms, or migration coverage with AST-aware queries.
- Use native search only for exact known paths/names and tests/logs only for runtime proof. Terminal grep or broad file dumps must not substitute for Codebase, Serena, or ast-grep.
- Research handoffs must record the exact Codebase project/index, Serena symbols, ast-grep patterns, evidence paths, gaps, and the next bounded change seam.
- Every research or code-changing Kanban card must include absolute paths to `AGENTS.md` and matching instructions, plus explicit measurable acceptance criteria for Codebase, Serena, and ast-grep evidence. A card without this routing is incomplete and must be corrected before dispatch.
- Notes and external model documentation may be read as source material by a dedicated subagent, but claims about this repository's architecture or impact must still be verified through the three code-intelligence routes.
- If one required tool is genuinely unavailable, record the exact failure and block the affected research lane; do not silently fall back to broad manual reading.
