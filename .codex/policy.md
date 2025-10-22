# Codex Architectural Ambition (Repo-wide)

**Goal:** maximize modularity, reuse, performance, maintainability.

**You MAY:**

- Create/rename/move modules/packages within a subfolder.
- Introduce interfaces via `typing.Protocol` or ABCs and invert dependencies.
- Extract pure functions, eliminate global state, reduce I/O coupling.
- Add docstrings, type hints (PEP 695), and tighten Ruff/mypy configs.
- Replace ad-hoc patterns with proven ones: ports & adapters, strategy, pipeline, plugin registry (pluggy), command/query split where natural.
- Add small unit tests for new seams and regression guards.

**You SHOULD:**

- Keep changes atomic, staged by passes.
- Preserve existing behavior unless RFC says otherwise.
- Leave TODOs for risky items; propose safe migration shims.

**You MUST NOT:**

- Touch files outside the target subfolder.
- Introduce external services or break public CLI/API without a migration section in the RFC.
