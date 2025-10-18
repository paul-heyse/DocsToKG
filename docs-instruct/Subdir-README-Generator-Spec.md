# Subdirectory README Authoring Guide for AI Agents
Last updated: 2025-10-18

> This guide is an instruction set for AI coding agents (e.g., GPT-based code assistants, Cody, Cursor, Claude Code) to create **subdirectory-scoped** `README.md` files that are safe, accurate, and highly machine-consumable. It upgrades the previous version by adding machine-readable **front matter**, a JSON **x-agent-map** appendix, explicit **Security & Data Handling**, and **Reliability (SLIs/SLOs)**—all tuned for agents to write correct, efficient code that leverages existing functionality.

---

## 1) Goal and Definition of Done (DoD)

**Goal:** Produce a concise but complete README **for one subdirectory** that enables a developer or agent to:
- Understand the submodule's purpose and boundaries.
- Set up and run it quickly (reproducibly).
- Modify it safely with clear invariants and contracts.
- Locate key files, schemas, and extension points without grepping the repo.
- See how it interacts with adjacent modules.

**Definition of Done (DoD):**
- README is created from **static analysis** of the subdirectory, with optional command outputs (only if allowed).
- Includes **YAML front matter** at the very top (see §2A) with ownership, stability, interfaces, versioning, last_updated, etc.
- Contains the **required sections** from §3, **in order**.
- Includes **two Mermaid diagrams** (one context/container-style flowchart, one sequence) that render on GitHub.
- Includes **Security & data handling** (ASVS level, threats considered, data classification) and **Observability** (SLIs/SLOs).
- Ends with a fenced JSON block named **x-agent-map** (see §2B) for machine parsing by tools.
- All paths are **relative to repo root** unless otherwise stated.
- No speculative claims; include `TODO:` markers for unknowns.
- Passes the linter rules in `Subdir-README-Lint.json` (front matter present; section presence & order; freshness; Mermaid count).

---

## 2) Inputs & discovery (how to gather facts)

You may have command execution enabled (Dev Containers, MCP tools) or not. Use *progressive discovery*:

### A. Static-only (always available)
- Read the subdirectory file tree.
- Parse code to extract:
  - CLI frameworks (argparse, Click, Typer) and their commands/flags.
  - Environment variables via `os.getenv(...)` / `os.environ.get(...)`.
  - Configuration models (e.g., Pydantic/dataclasses) and defaults.
  - Public entry points (`__main__`, `__init__`, CLI/HTTP/event handlers).
  - Schemas/contracts (`schema/*.json`, `openapi*.yaml`, protobufs).
  - Observability hooks (OpenTelemetry/logging/metrics emission).
- Inspect `tests/` for typical workflows and invariants.
- Discover repo-level metadata where present: `CODEOWNERS`, `SECURITY.md`, SBOM, `devcontainer.json`, `justfile`.

### B. Optional runtime discovery (only if allowed)
- `just --list` or `make help` to enumerate tasks.
- `python -m <package> --help` to capture CLI surface.
- `pytest -q <subdir-tests>` to confirm test/discovery entry points.
- Any project-specific doctor/health command (e.g., `... doctor`).

### C. Upstream/downstream mapping
- Search for imports from sibling packages (`from ..<module> import ...`).
- Identify outputs (artifacts, reports) consumed by other subfolders.
- Record any ID/path invariants and where they are enforced.

> If data is missing, include a `TODO:` line and keep moving—do not invent details.

---

## 2.5) README front matter (machine-readable)

At the very top of the README, before the title, add YAML front matter with these keys (expand as needed):

```yaml
---
subdir_id: <stable-id>
owning_team: <org/team>
interfaces: [cli, http, events]
stability: stable|beta|experimental
versioning: semver|date-based|none
codeowners: <@team-or-people>
last_updated: 2025-10-18
related_adrs: [ADR-0003, ADR-0012]
slos:
  availability: "99.9%"
  latency_p50_ms: 120
data_handling: pii|no-pii
sbom:
  path: <path-to-spdx-or-cyclonedx>
---
```

> **Purpose:** allows agents/tools to consume ownership, stability, compatibility, and compliance metadata without scraping prose.

---

## 2.6) x-agent-map appendix (JSON, machine-readable)

At the very end of the README, include a fenced block named **x-agent-map** with minimally these fields:

```json
{
  "entry_points": [{"type":"cli","module":"<pkg.cli>","commands":["run","doctor"]}],
  "env": [{"name":"FOO_API_URL","default":"http://localhost:8080","required":true}],
  "schemas": [{"kind":"openapi","path":"api/openapi.yaml"}],
  "artifacts_out": [{"path":"out/*.json","consumed_by":["../analytics"]}],
  "danger_zone": [{"command":"just wipe-cache","effect":"deletes ./cache"}]
}
```

> **Purpose:** gives agents a reliable, parseable contract surface (entry points, env, schemas, artifacts, destructive commands).

---

## 3) Output: required sections (in this order)

1. **Title**  
   `# <Repo Name> • <Subdirectory Name>`

2. **Purpose & scope boundary** *(explanation)*  
   One or two sentences. Include a one-line boundary: what this submodule does **and does not** do.

3. **Quickstart** *(how-to)*  
   Shortest path to run the core workflow (prefer Dev Container; provide local alternative).

4. **Common commands** *(reference)*  
   - Prefer `just`/`make` task names if present (self-documenting).
   - Provide direct CLI alternatives (`python -m package ...`).

5. **Folder map (top N)** *(reference)*  
   8–15 lines max. For each key folder/file: a one-line purpose (no prose).

6. **System overview** *(explanation)*  
   Two small Mermaid diagrams: a **context/container-style flowchart** and a **sequence** for a common request path.
   - **Diagram DoD:** mark external dependencies, trust boundaries, and one failure/retry path.

7. **Entry points & contracts** *(reference)*  
   - CLI/HTTP/event entry points and the relevant files.
   - Invariants and layering rules (e.g., pure domain layer, single I/O path).
   - Interfaces or protocols implemented (e.g., Provider protocol).

8. **Configuration** *(reference)*  
   - Env vars with defaults and meaning (`.env.example` if present).
   - Config validation command (if available).

9. **Data contracts & schemas** *(reference)*  
   - Paths to JSON Schemas, OpenAPI, protobufs, etc.
   - How to validate or generate them.

10. **Interactions with other packages** *(explanation/reference)*  
    - Upstream inputs (if any).
    - Downstream consumers and guarantees (IDs, paths, formats).

11. **Observability** *(reference)*  
    - Logs/metrics/tracing and how to view them (link to dashboards if any).
    - **SLIs/SLOs**: list 2–3 SLIs and their SLO targets (e.g., success rate, P50/P95 latency).
    - Health/doctor commands and expected exit codes.

12. **Security & data handling** *(reference/explanation)*  
    - **ASVS** level (L1/L2/L3) relevant to this module.
    - Top **STRIDE** threats considered (3–5 bullets) and mitigations or upstream responsibilities.
    - Data classification (PII/no-PII); secrets handling rules.
    - Any relevant secure coding invariants (e.g., constant-time compare for secrets).

13. **Development tasks** *(how-to/reference)*  
    - Lint/typecheck/test commands.
    - Typical inner loop.
    - Pre-commit hooks.

14. **Agent guardrails** *(reference)*  
    - What agents **may/must not** change (hashing, IDs, schema field order, versioning schemes).
    - **Danger zone**: destructive commands (e.g., cache wipes, migrations).

15. **FAQ** *(explanation)*  
    - 3–8 frequent questions with crisp answers.

> *Optional:* A short **Changelog hook** at the top can link to `CHANGELOG.md` if the subdir has independent releases.

---

## 4) Style & formatting rules

- Use Markdown H1..H3 headings with stable anchor names; **Sentence case** headings.
- Keep bullets tight; avoid paragraphs over 5 lines.
- Prefer ASCII for portability (avoid smart quotes/arrows).
- Use fenced code blocks for commands; annotate language (bash, json, yaml).
- Keep the **Folder map** to 8–15 items; link out for details.
- Put placeholders in `ALL_CAPS_LIKE_THIS` or `<LIKE_THIS>`.
- Use relative links, e.g., `../OtherModule/README.md`.
- **Mermaid diagrams** must be small and high-signal; use GitHub-native Mermaid.
- If a **Justfile** exists, mirror CLI commands as recipes and reference `just --list`.

---

## 5) Reliability, security & correctness guardrails

- Treat all external input as untrusted; call this out in **Security & data handling**.
- Avoid executing downloaded artifacts or external code.
- Reflect package boundaries that prevent cross-module writes.
- Mention destructive commands in **Danger zone**.
- Record 2–3 SLIs/SLOs and link to dashboards when available.
- If you cannot verify a fact, add `TODO:` rather than guessing.

---

## 6) Algorithm to generate the README (step-by-step)

1. Identify subdirectory root `SUBDIR` and repo root `ROOT`.
2. Build a file tree and pick 8–15 high-signal files/dirs for the **Folder map**.
3. Extract entry points: scan for CLI/HTTP/event handlers and main functions.
4. Extract env vars and config models; capture defaults if obvious.
5. Locate schemas/contracts and write short descriptions.
6. Map upstream/downstream interactions via imports and output paths.
7. Draft the two **Mermaid** diagrams (flowchart + sequence) with trust boundary and one failure path.
8. Write the **Required sections** in order, filling facts collected.
9. Insert **YAML front matter** (§2.5) at the very top (before the title).
10. Append **x-agent-map** (§2.6) JSON block at the very end.
11. Insert `TODO:` markers for anything unknown.
12. Update `last_updated` to today’s date.
13. Run the linter spec from `Subdir-README-Lint.json` and fix issues.
14. Save as `SUBDIR/README.md` with Unix newlines.

---

## 7) Acceptance checklist (for humans/agents)

- [ ] YAML front matter exists and includes required keys; `last_updated` ≤ 90 days old.
- [ ] All sections from §3 exist **in order**.
- [ ] Two Mermaid diagrams render on GitHub.
- [ ] Commands are runnable or explicitly marked as examples.
- [ ] Paths are relative to repo root and correct.
- [ ] Security section present (ASVS level, threats, data classification).
- [ ] Guardrails list do/do-not and danger zone.
- [ ] File passes the linter JSON checks (including link and Mermaid syntax checks, if enabled).

---

## 8) Stub templates you can reuse

See `Subdir-README-Template.md` for a fill-in skeleton (kept in sync with this guide).
