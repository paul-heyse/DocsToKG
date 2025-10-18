# Subdirectory AGENTS.md Authoring Guide (Quality & Performance Focus)

Last updated: 2025-10-18

This guide instructs AI coding agents to produce **subdirectory-scoped** `AGENTS.md` files that
maximize **functional correctness, performance, modularity, and documentation quality**.
It intentionally **omits budgets, allowlists, or operational restrictions**. Pair this with our
Module Organization, Code Annotation, and Style guides when drafting recommendations and examples.

---

## 1) Goal and Definition of Done

**Goal:** Provide a pragmatic, high-signal agent runbook for one subdirectory (SUBDIR) that helps
authors ship **more functional, faster, and cleaner code** while keeping documentation first-class.

**Definition of Done (DoD)**

- The file is specific to the SUBDIR and contains the **required sections** in §3, in order.
- Guidance is **actionable** (commands, patterns, examples) and tied to on-disk paths.
- The doc links to or embeds **code documentation guidance** (docstrings, NAVMAP, style).
- Performance advice includes **hot paths, baselines, and optimization playbook**.
- Commands are runnable or labeled as examples.
- All claims are factual; use `TODO:` for unknowns.
- Uses **Mermaid** for one architecture or flow diagram (small and precise).

---

## 2) Inputs & Discovery (how to gather facts)

Use *progressive discovery*, prefer static analysis first:

### A. Static-only (always available)
- Walk the SUBDIR; map **entry points**, public APIs, core classes/functions.
- Parse code for **tight loops**, large comprehensions, recursive patterns, repeated I/O,
  regex hot spots, `json.dumps/loads` churn, and potential **N^2 joins/scans**.
- Identify **data shapes** (schemas, Pydantic models, typed dicts) and typical sizes.
- Locate **caching** and memoization (e.g., `functools.lru_cache`, local caches).
- Inspect tests for **benchmarks**, large fixtures, and integration paths.
- Check docs: **NAVMAP** blocks, module docstrings, and example usage.
- Note places where **vectorization** (NumPy/Pandas), **batching**, or **streaming** could apply.

### B. Optional runtime probes (only if explicitly allowed by the environment)
- Run **micro-benchmarks** or representative tests (e.g., `pytest -q -k perf -q`).
- Quick profile with `python -m cProfile -m <pkg.module> <args>` or PyInstrument if present.
- Capture **before/after** timing for one hot path as a baseline.

> If anything cannot be verified, emit `TODO:` with a short note (do not invent numbers).

---

## 3) Output: required sections (in this exact order)

1. **Mission and Scope** — What excellence looks like for this SUBDIR (functional goals + boundaries).
2. **High-Level Architecture & Data Flow** — One Mermaid diagram (flowchart or sequence) plus 3–6 bullets.
3. **Hot Paths & Data Shapes** — The functions/methods most executed; typical input sizes and shapes.
4. **Performance Objectives & Baselines** — Targets (e.g., P50 latency, throughput) and any known baselines.
5. **Profiling & Optimization Playbook** — How to measure; common tactics (batching, vectorization, caching, avoiding N^2).
6. **Complexity & Scalability Guidance** — Expected Big‑O for key ops; memory growth notes; large‑N strategies.
7. **I/O, Caching & Concurrency** — Disk/network patterns, cache keys/invalidation, safe parallelism.
8. **Invariants to Preserve (change with caution)** — Contracts that keep behavior correct (ordering, idempotency, stable IDs).
9. **Preferred Refactor Surfaces** — Files/regions that are designed to be extended or swapped to improve quality or speed.
10. **Code Documentation Requirements** — What to document and how (docstrings, NAVMAP, examples, cross‑refs).
11. **Test Matrix & Quality Gates** — Lint/typecheck/tests/benchmarks to run; minimum expectations.
12. **Failure Modes & Debug Hints** — Short table mapping symptom → likely cause → quick checks.
13. **Canonical Commands** — `just`/`make` tasks and CLI equivalents for build, test, profile, and docs.
14. **Indexing Hints** — Where newcomers (or tools) should start reading; highest-signal files.
15. **Ownership & Documentation Links** — Pointers to CODEOWNERS (if any) and local docs.
16. **Changelog and Update Procedure** — When and how to update this file.

---

## 4) Section-by-section authoring guidance

### 4.1 High‑Level Architecture & Data Flow
- Keep the diagram small: main components, primary data edges, and one failure path.
- Prefer **C4‑style** thinking, but write in Mermaid for GitHub rendering.
- Call out **boundaries** (pure domain logic vs I/O), which unlock safe refactors.

### 4.2 Hot Paths & Data Shapes
- Name the hot path functions/methods and **why** they’re hot (e.g., called in a loop).
- Capture typical **batch sizes**, average/95th **payload sizes**, and critical **schema fields**.

### 4.3 Performance Objectives & Baselines
- If historic metrics exist, summarize them; otherwise, propose **sensible targets** and mark them `TODO`.
- For each hot path, give a simple **measurement recipe** (command + expected output).

### 4.4 Profiling & Optimization Playbook
- Include **one-liners** for cProfile/Pyinstrument/pytest-benchmark (if present).
- Recommend **batching**, **vectorization**, reducing **allocations**, and moving **I/O off hot paths**.
- Note any known **contention points** (GIL-sensitive sections, shared caches).

### 4.5 Complexity & Scalability Guidance
- State the **expected complexity** of key operations; warn about **quadratic joins/scans**.
- Provide **large‑N** strategies (e.g., chunked processing, merge‑sort style batching).

### 4.6 I/O, Caching & Concurrency
- Show where **caching is effective** (and invalidation rules).
- Document **streaming reads/writes**, **memory‑mapped** options, or **async** boundaries.

### 4.7 Invariants to Preserve
- List invariants around **ordering**, **idempotency**, **hash/ID computation**, and **schema field order**.
- These are not “forbidden changes”, but note the **risk** of violating them.

### 4.8 Preferred Refactor Surfaces
- Point to **extension points** (adapters, providers, strategy classes) designed for change.
- Mention files where refactors have **lowest regression risk**.

### 4.9 Code Documentation Requirements
- Require **module‑level docstrings**, **public API docstrings**, and **NAVMAP** blocks in large modules.
- Align with our **Module Organization** (ordering, dividers, NAVMAP) and **Style** guides.
- Include 1–2 **worked examples** (short runnable snippets) for key entry points.

### 4.10 Test Matrix & Quality Gates
- Specify commands for **lint**, **typecheck**, **unit/integration tests**, and (if available) **benchmarks**.
- Encourage table‑driven tests, **golden files** for transformations, and stress tests for hot paths.

---

## 5) Style & formatting rules

- Mirror the section titles verbatim to satisfy lint checks.
- Use sentence‑case headings; keep bullets short and concrete.
- Prefer fenced code blocks with language tags (bash, python, json, yaml).
- Keep the **Failure Modes** table to 3–10 rows of the highest‑impact cases.
- Use ASCII; no smart quotes/arrows.

---

## 6) Algorithm to generate AGENTS.md

1. Identify SUBDIR root.
2. Collect facts via §2A; only run probes from §2B if explicitly allowed.
3. Draft each section in §3, in order; **diagram** in §2; add examples.
4. For hot paths, include at least **one** feasible measurement recipe.
5. Cross‑check examples: make sure commands exist or mark them as examples.
6. Insert `TODO:` for unknown values (e.g., baselines) rather than guessing.
7. Link to Module Organization Guide, Code Annotation Standards, and Style Guide.
8. Save as `SUBDIR/AGENTS.md` and run the linter.

---

## 7) Acceptance checklist

- [ ] Sections 1..16 present and ordered.
- [ ] Contains a Mermaid diagram in Architecture & Data Flow.
- [ ] Hot paths and data shapes documented.
- [ ] Profiling/optimization playbook includes runnable commands or marked examples.
- [ ] Invariants listed with rationale; preferred refactor surfaces identified.
- [ ] Code documentation guidance ties back to NAVMAP and docstring standards.
- [ ] Test matrix present and actionable.
- [ ] File passes linter checks.
