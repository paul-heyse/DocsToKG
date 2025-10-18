# Agents Guide - OntologyDownload

Audience: AI software engineering agents (Cursor/Cody/Claude Code/etc.) and human operators working only in `src/DocsToKG/OntologyDownload`.  
Last updated: 2025-10-18

---

## Mission

Implement, refactor, and test the ontology downloader while preserving determinism, safety, and package boundaries. This package only plans, fetches, and validates artifacts. It does not write to vectors or graph stores.

---

## Canonical commands

```bash
# Bootstrap
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"

# Planning -> pulling -> validating
python -m DocsToKG.OntologyDownload plan     --sources ./configs/ontology-sources.yaml --out ./artifacts/plan.jsonl
python -m DocsToKG.OntologyDownload pull     --plan ./artifacts/plan.jsonl             --out ./data/ontologies
python -m DocsToKG.OntologyDownload validate --in   ./data/ontologies                  --report ./artifacts/validation-report.jsonl

# Diagnostics
python -m DocsToKG.OntologyDownload doctor

# Tests
pytest -q tests/ontology_download
```

If present, prefer `just` tasks (they are the source of truth for routine flows):

```bash
just ontology.plan
just ontology.pull
just ontology.validate
just ontology.doctor
```

---

## Boundaries & invariants (do not break)

- No cross-package writes. Do not touch vector stores (FAISS) or Neo4j from here. Output is only: `plan.jsonl`, downloaded artifacts, and `validation-report.jsonl`.
- Deterministic planning. Same inputs -> same `plan.jsonl` ordering and contents. If you change sorting/selection logic, update the JSON Schema and tests.
- Single I/O path. All HTTP/network traffic must go through `io/http.py` (retries, timeouts, rate limits). Do not embed `requests`/`httpx` elsewhere.
- Integrity first. Every artifact gets a sha256 and stable cache path. Do not bypass hashing or write files outside `data/ontologies/`.
- Pure planning. `planning/` is side-effect free. No filesystem or network writes there.
- Validators are idempotent. Re-running `validate` over identical inputs must produce identical outputs.

---

## Safe editing surfaces

- `planning/*` — planning rules, selectors, sort order, plan item shape
- `resolvers/*` — add/extend provider adapters (implement `plan`/`fetch`)
- `validation/*` — RDF/JSON-LD parsing, SHACL-lite checks, size/MIME checks
- `settings/*` — add configuration with sensible defaults (document env vars)
- `io/*` — improve caching, decompression, content-type sniffing

Avoid / Do not edit:
- Shared constants that define hashing/caching layout without updating migration notes + tests.
- Any downstream package folders (DocParsing, HybridSearch, KnowledgeGraph).

---

## How to add a provider (step-by-step)

1) Scaffold
- Create `resolvers/<provider>/adapter.py` with `plan(ctx)` and `fetch(item, ctx)`.
- Register the provider in `resolvers/__init__.py` or via the plugin registry.

2) Plan
- Accept a provider section in `configs/ontology-sources.yaml`.
- Emit deterministic `PlanItem`s with `source_id`, `url`, `format`, `license`, `target_relpath`, optional `expected_sha256`.

3) Fetch
- Use `io/http.py` helpers only. Respect `HTTP_TIMEOUT_S`, `RETRY_MAX`, `RATE_LIMIT_TOKENS_PER_MIN`.
- Return bytes + metadata; the caller handles hashing and writes.

4) Validate
- Ensure the artifact parses as RDF (RDF/XML, Turtle, JSON-LD) if applicable.
- Add provider-specific warnings (license mismatch, missing version header, etc.).

5) Test
- Add fixtures under `tests/fixtures/<provider>/`.
- Write `tests/ontology_download/test_<provider>.py` (plan, fetch, validate).

6) Docs
- Update `configs/ontology-sources.yaml` and the package README with any nuances.

---

## Common tasks

- Refactor planning rules (e.g., sort by `priority` then `source_id`).
- Add ZIP/TAR handling (ensure no path traversal; flatten to normalized filenames).
- Introduce rate limits per provider (token bucket in `io/http.py`).
- Strengthen SHACL-lite checks (presence of classes/properties as minimal structure).
- Improve diagnostics (more reason codes in `validation-report.jsonl`).

---

## Failure modes & fixes

- HTTP 429/5xx storms -> Verify `RETRY_MAX` and backoff jitter; check `RATE_LIMIT_TOKENS_PER_MIN`.
- Non-deterministic plans -> Ensure sort keys are stable (`source_id`, `target_relpath`).
- RDF parse errors -> Capture content type and first 256 bytes in the report (no secrets). Add a provider-specific fix or fallback format.
- Archive traversal -> Reject members with `..` or absolute paths; normalize to ASCII-safe filenames.
- Large files -> Stream to disk via `io/http.py`; avoid loading the entire payload into memory.

---

## Security & safety

- Treat downloaded content as untrusted. Never execute. Never import as code.
- Enforce content-type, size limits, and sha256 checks before saving.
- Sanitize all filenames; block path traversal on extraction.
- Do not run shell commands not present in `just`/README without explicit approval.

---

## Test matrix (what to run before opening a PR)

```bash
just lint
just typecheck
pytest -q tests/ontology_download
python -m DocsToKG.OntologyDownload doctor
```

If `just` is unavailable, use equivalent `make` targets or bare commands.

---

## PR checklist (subfolder-scoped)

- [ ] Updated/added unit tests under `tests/ontology_download/`
- [ ] No changes to hashing/caching layout without migration notes
- [ ] Deterministic plan verified on the same inputs (CI job)
- [ ] New env vars documented in `README.md` (`.env.example` updated if used)
- [ ] Provider added to registry; sources config updated
- [ ] Logs are structured; no secrets printed

---

## Indexing hints for agents

- Read `schema/*.schema.json` to understand `plan.jsonl` and `validation-report.jsonl`.
- Prefer searching by identifiers (`source_id`, `target_relpath`) rather than free text when editing planners/validators.
- High-signal files: `cli.py`, `planning/*`, `resolvers/*`, `validation/*`, `io/http.py`, `settings/*`.

---

## Contact & ownership

If this subfolder has code owners, they will be listed in the root `CODEOWNERS` (look for a line like `src/DocsToKG/OntologyDownload/ @owner1 @owner2`).

---

## Keep this guide current

When you change a public contract (schemas, CLI flags, directory layout), update both `README.md` (overview) and this `AGENTS.md` (execution details).
