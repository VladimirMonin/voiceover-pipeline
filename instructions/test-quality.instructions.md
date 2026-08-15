---
name: Test quality contract
description: Deterministic test expectations and verification tiers for Python and CLI changes.
applyTo: "{src/**/*.py,tests/**/*.py,pyproject.toml,uv.lock}"
---

# Test quality contract

- Add or update tests for observable behavior, regressions, error paths, output safety, and JSON/exit-code compatibility.
- Keep tests deterministic and offline: use temporary directories, fixtures, and mocks; never depend on `.env`, real credentials, cloud providers, paid APIs, GPU availability, or external downloads.
- Use synthetic secret placeholders only and assert that diagnostics do not leak them where relevant.
- Run the narrowest relevant test selection first, then the broader suite when scope permits. Do not mutate dependencies or `uv.lock` merely to run tests.
- If the task is documentation/instructions-only, do not run application tests unless requested; validate structure, links/frontmatter, and `git diff --check` instead.
- Never encode a current test count or transient historical result in permanent instructions.
- Report commands and actual outcomes; clearly state skipped checks and blockers.
