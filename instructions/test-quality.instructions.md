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
- For every Python code change, run `uv run ruff check src tests`, `uv run ruff format --check src tests`, and `uv run mypy --no-incremental`. These are required verification gates, not optional diagnostics. `--no-incremental` avoids stale compiled mypy cache crashes after tool upgrades.
- Implementers may run `uv run ruff check --fix <touched paths>` and `uv run ruff format <touched paths>` only on files in the current change. Do not reformat unrelated historical files merely to make the whole-tree formatter green.
- Reviewers rerun Ruff lint, Ruff formatting, and mypy on the exact reviewed bytes. Fix ordinary one-to-three-line lint, formatting, import, annotation, fixture, or stale-assertion defects directly in the current review scope and rerun the smallest affected check; do not create a repair card for such residue. Material behavior, contract, safety, or architecture defects still require rejection and bounded repair.
- If a whole-tree Ruff or mypy baseline is already red outside the changed paths, prove the changed paths are clean, record the exact pre-existing failures, and do not weaken configuration or block the scoped change without evidence that it caused them.
- If the task is documentation/instructions-only, do not run application tests unless requested; validate structure, links/frontmatter, and `git diff --check` instead.
- Never encode a current test count or transient historical result in permanent instructions.
- Report commands and actual outcomes; clearly state skipped checks and blockers.
