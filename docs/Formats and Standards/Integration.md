Absolutely—here’s a repo-ready Markdown you can drop in as `docs/INTEGRATION-REVIEW.md`. I also included a one-liner to add/commit it.

---

# DocsToKG — Cross-Subsystem Integration Review

**Generated:** 2025-10-23

This document captures the cross-cutting integrations and contracts across **ContentDownload**, **DocParsing**, **HybridSearch**, and **OntologyDownload**. It consolidates decisions and highlights risks, guardrails, and next actions so that engineering and SRE have a single, current reference.

---

## 1) System Overview (who produces → who consumes)

**A. IDs & provenance (the spine)**

* `doc_id` is created in **ContentDownload** and preserved by **DocParsing** (DocTags → Chunks → Embeddings).
* **HybridSearch** indexes **`uuid` per chunk** (deterministic from `doc_id + chunk_idx`) and carries `doc_id` in metadata for shaping, joins, and audits.
* This spine enables **resume**, **audits**, and **fusion diagnostics**.

**B. Artifacts → parsing**

* **ContentDownload** normalizes source assets (PDF/HTML/XML) and writes a **manifest + SQLite catalog**.
* **DocParsing** consumes these artifacts and produces **DocTags, Chunks, and Embeddings** with **append-only manifests + `manifest.sqlite3`** and per-stage provenance.

**C. Embeddings → retrieval**

* **DocParsing** outputs **Parquet (canonical)** for vectors (dense/sparse/lexical) in a fixed partition layout with footer metadata.
* **HybridSearch** ingests Parquet by default (JSONL accepted but not preferred) and enforces **strict UUID ↔ FAISS-id bijection** during add/train/migrate.

**D. Ontologies → KG gates (adjacent to retrieval)**

* **OntologyDownload** ensures **reproducible ontology corpora** (DuckDB catalog + lockfiles) and **strict validators** (pySHACL etc.).
* These ontologies are used to gate Knowledge Graph releases and enable concept mapping & SHACL conformance, then surface next to HybridSearch answers.

**E. Observability**

* Metrics families: `content_*`, `docparse_*`, `hybrid_*`, `ontofetch_*` + JSONL logs + OTel spans.
* Labels standardized across stages: `run_id`, `config_hash`, `namespace`, `model_id`, and **backend** (`cuvs|faiss`).

---

## 2) Integration Contracts (binding)

### A. Identifier & ordering invariants

* **Deterministic UUIDs** in DocParsing (e.g., `uuid = f(doc_id, chunk_idx)`).
* **HybridSearch strict mode by default:** `vector_cache_limit = 0` → **fail fast** on first UUID/row drift.
* Result: **UUID ↔ FAISS-id bijection** is guaranteed across build, snapshot, and restore.

### B. Vector format & schema

* **Parquet is canonical** (`Embeddings/{family=dense|sparse|lexical}/fmt=parquet/YYYY/MM/<doc_id>.parquet`).
* Parquet **footer keys** (required): `docparse.provider`, `model_id`, `dtype`, `cfg_hash`, and **`vector_schema_version`** (new).
* JSONL is **supported** for compatibility; not preferred for large batches.

### C. Resume & strict modes

* **ContentDownload:** idempotent resume from JSONL/CSV/SQLite.
* **DocParsing:** fast resume by `doc_id`; `--verify-hash` for strict provenance; fingerprints (`.fp.json`) across stages.
* **HybridSearch:** strict UUID alignment (`vector_cache_limit=0`); ingestion sorts by `uuid` and asserts **dimension** and **footer** invariants.
* **OntologyDownload:** strict mode **purges staged artifacts** upon validator failure; DB-backed **plan caching is mandatory**.

### D. Retrieval & fusion defaults

* **Index policy (binding):** Flat **< 1M**, IVFFlat **≥ 1M**, IVFPQ when memory-bounded.
* **cuVS preferred:** `use_cuvs: auto` → pick cuVS when compiled; **FAISS fallback** is transparent.

  * **1–10M** → cuVS **IVF-Flat**; **≥10M** → cuVS **CAGRA** (or cuVS **IVF-PQ** if memory-bounded).
  * **FP16 OFF by default**; 1 GiB FAISS scratch; replication only if configured.
* **Fusion:** **RRF** weights `{bm25: 0.35, splade: 0.0, dense: 0.65}`, `k0 = 60`; **MMR enabled**, `λ = 0.7`.

### E. Security & politeness

* **No raw dense vectors** returned by APIs; only UUIDs/scores/metadata.
* **HTTPS + allowlists**; **Tenacity** backoff honoring **`Retry-After`**; robots obeyed by default; **Hishel** RFC-9111 HTTP caching.

---

## 3) Integration Risks & Guardrails

### 3.1 Embedding dimension drift

* **Risk:** Changing the dense model silently changes vector dim; ingestion breaks.
* **Guardrail:** Ingest checks **Parquet footer** (`model_id`, `dtype`) and **dimensions**; rejects mismatches with actionable logs/metrics.

### 3.2 UUID/row misalignment

* **Risk:** Mixed JSONL/Parquet or non-stable sorting breaks bijection.
* **Guardrail:** Production path **mandates Parquet**; ingestion **sorts by `uuid`**; `vector_cache_limit=0` in prod; CI test ingests shuffled input and asserts bijection.

### 3.3 Parquet schema evolution

* **Risk:** Changing columns/footers breaks older readers.
* **Guardrail:** Introduce **`vector_schema_version`** (footer). HybridSearch accepts `vN` with explicit migrations and defaults.

### 3.4 cuVS/FAISS parity regressions

* **Risk:** Faster index regresses quality (nDCG).
* **Guardrail:** Add `backend` label (`cuvs|faiss`) to `hybrid_*` metrics; run a **canary non-regression gate** (route namespace to FAISS if cuVS degrades **>1% nDCG@10**).

### 3.5 Cross-stage correlation

* **Risk:** Hard to connect slow queries to the exact content/parse run.
* **Guardrail:** Standard **labels** across metrics/logs: `run_id`, `config_hash`, `namespace`, `model_id`. Dashboards link from `hybrid_query_latency_seconds` to DocParsing manifests.

### 3.6 Config prefix drift

* **Risk:** Mixed `DTKG_*` vs `DOCSTOKG_*`.
* **Guardrail:** Prefer `DOCSTOKG_*`; still read `DTKG_*`, but emit a **deprecation notice** in `print-config`.

### 3.7 Snapshot cadence vs. consistency

* **Risk:** Frequent partial snapshots expose inconsistent top-K.
* **Guardrail:** Enforce **time + write thresholds** for snapshotting and **atomic swap** after UUID-stable add+train+migrate.

### 3.8 Politeness duplication

* **Risk:** Divergent implementations of backoff/headers per stage.
* **Guardrail:** Extract a **shared “Polite HTTP”** module (Tenacity, allowlists, robots, Hishel) used by ContentDownload and OntologyDownload.

### 3.9 KG acceptance gates

* **Risk:** Missing hard contract for DocParsing→KG concept mapping.
* **Guardrail:** Define a **mapping staging schema** (`Mention`, `Link`, `Triple`) keyed by `doc_id/uuid` and IRI; enforce **pySHACL=0** before KG publish.

---

## 4) Stage Acceptance Gates (rollup)

* **ContentDownload:** Yield ≥ **85%**, `429 ratio` < **1%**, TTFP p50 ≤ **3s**, p95 ≤ **10s** (per source).
* **DocParsing:** Parse success ≥ **98%**; title/abstract fidelity ≥ **99%**; **dim-mismatch = 0**; stable `cfg_hash`/`__config__` written.
* **HybridSearch:** Retrieval p50 ≤ **150 ms**, p99 ≤ **600 ms**; hybrid uplift ≥ **+5% nDCG@10** vs best single retriever; UUID bijection preserved.
* **OntologyDownload:** **pySHACL = 0**; ROBOT profile ✓; Arelle ✓; **plan caching mandatory**; strict mode purges failed artifacts.

---

## 5) Quick Wins

1. **Footer contract:** add `vector_schema_version` and assert footers at ingest; log diffs.
2. **Ingest guardrail:** set `--vector-cache-limit 0` in prod; CI test for shuffled-input bijection.
3. **Metric labels:** add `backend` (cuvs|faiss), `model_id`, `namespace`, `config_hash` to all metric families.
4. **Env prefix:** deprecate `DTKG_*` in `print-config`; recommend `DOCSTOKG_*` in docs.
5. **cuVS canary:** implement the >1% nDCG non-regression gate and auto-route fallback.

---

## 6) Next-Action Mini-PRs

* **HybridSearch**

  * Enforce **dimension + footer** checks on ingest (reject mismatches).
  * Implement **`backend` label** and **non-regression gate** (cuVS↔FAISS).
  * Default **Parquet ingestion** path; sort by `uuid`; `vector_cache_limit=0` in prod configs.

* **DocParsing**

  * Ensure Parquet footer includes **`vector_schema_version`** and **`cfg_hash`**.
  * Verify `uuid = f(doc_id, chunk_idx)` is deterministic across chunker modes.

* **ContentDownload**

  * Factor shared **Polite HTTP** module (Tenacity, allowlists, robots, Hishel).
  * Emit `model_id`/`namespace` labels where applicable (for end-to-end joins).

* **OntologyDownload**

  * Confirm **pySHACL** is registered/enabled; keep **plan caching mandatory**.
  * Add `db backup` CLI and metrics shim (`ontofetch_*`) if not present.

---

## 7) Appendix — Canonical Paths & Keys

**Vectors (Parquet)**

```
Embeddings/{family=dense|sparse|lexical}/fmt=parquet/YYYY/MM/<doc_id>.parquet
```

**Parquet footer keys (required)**

```
docparse.provider
model_id
dtype
cfg_hash
vector_schema_version
```

**HybridSearch defaults**

```
use_cuvs: auto
index policy: <1M → FAISS Flat; 1–10M → cuVS IVF-Flat; ≥10M → cuVS CAGRA
FP16: off
RRF: {bm25: 0.35, splade: 0.0, dense: 0.65}, k0=60
MMR: lambda=0.7
```

---

*This document is the integration source of truth for the current milestone. Update it whenever cross-stage contracts change.*

---

## Add to repo

```bash
# from repo root
mkdir -p docs
printf "%s\n" '<<PASTE CONTENT ABOVE>>' > docs/INTEGRATION-REVIEW.md
git add docs/INTEGRATION-REVIEW.md
git commit -m "docs: add cross-subsystem INTEGRATION-REVIEW (contracts, risks, guardrails, PR actions)"
```

If you’d rather I open a branch + PR for this file, I can do that too—just say the word and tell me your preferred branch name (e.g., `docs/integration-review`).
