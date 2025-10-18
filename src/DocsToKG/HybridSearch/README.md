---
subdir_id: docstokg-hybrid-search
owning_team: TODO_TEAM
interfaces: [python, service]
stability: beta
versioning: semver
codeowners: TODO_CODEOWNERS
last_updated: 2025-10-18
related_adrs: []
slos: { availability: "TODO", latency_p50_ms: TODO }
data_handling: no-pii
sbom: { path: TODO_SBOM_PATH }
---

# DocsToKG • HybridSearch

Purpose: Hybrid retrieval engine combining lexical and dense vector search with configurable fusion and observability.
Scope boundary: Ingests chunked documents, maintains FAISS/OpenSearch-style indexes, and exposes search APIs; does not train embedding models or manage downstream ranking pipelines.

---

## Quickstart
> Bootstrap environment, then ingest a toy dataset and issue a hybrid search.
```bash
./scripts/bootstrap_env.sh
direnv allow                     # or source .venv/bin/activate
direnv exec . python examples/hybrid_search/ingest.py  # TODO: confirm script path
direnv exec . python examples/hybrid_search/query.py "hybrid search"
```

## Common commands
```bash
# Discover just tasks (if present)
just --list

# Typical workflows (TODO replace placeholders)
just hybrid.ingest
just hybrid.search
just hybrid.tests

# Fallback Python entry points
direnv exec . python -m DocsToKG.HybridSearch.devtools.ingest --config configs/hybrid-search.yaml
direnv exec . python -m DocsToKG.HybridSearch.service --serve   # TODO: confirm service CLI
```

## Folder map
- `service.py` – Hybrid search orchestration API (validation, fusion, pagination guards).
- `pipeline.py` – Chunk ingestion, feature generation, observability helpers.
- `store.py` – FAISS/OpenSearch adapters, vector math utilities, snapshotting.
- `config.py` – Pydantic-style configuration models for indexing, fusion, retrieval.
- `interfaces.py` – Protocol definitions for dense and lexical adapters.
- `router.py` – Namespace-aware FAISS router with snapshot/restore.
- `types.py` – Typed data contracts (`DocumentInput`, `HybridSearchRequest/Response`).
- `features.py` – Tokenization, sliding window, and feature generation primitives.
- `devtools/` – Simulation utilities (OpenSearch simulator, test harness helpers).
- `tests/hybrid_search/` – End-to-end and scale test suites.

## System overview
```mermaid
flowchart LR
  A[Documents / embeddings] --> B[ChunkIngestionPipeline]
  B --> C[ChunkRegistry / LexicalIndex]
  B --> D[ManagedFaissAdapter]
  subgraph Query
    Q[HybridSearchAPI] --> S[HybridSearchService]
    S --> C
    S --> D
    S --> F[ReciprocalRankFusion + MMR]
    F --> R[HybridSearchResponse]
  end
  D -.-> G[(FAISS snapshots)]:::boundary
  classDef boundary stroke:#f00;
  D:::boundary
```
```mermaid
sequenceDiagram
  participant U as Caller
  participant API as HybridSearchAPI
  participant SVC as HybridSearchService
  participant LEX as LexicalIndex
  participant DENSE as DenseVectorStore
  participant FUSE as Fusion
  U->>API: POST /hybrid-search (query, filters)
  API->>SVC: validate + route request
  SVC->>LEX: search_bm25 / search_splade
  SVC->>DENSE: search_many (FAISS)
  LEX-->>SVC: sparse hits
  DENSE-->>SVC: dense hits
  SVC->>FUSE: fuse + diversify + shape
  FUSE-->>API: ranked HybridSearchResponse
  API-->>U: JSON payload
```

## Entry points & contracts
- Entry points: Python APIs (`HybridSearchService`, `HybridSearchAPI`), ingestion helpers in `pipeline.py`, FAISS router/service glue in `router.py`.
- Contracts/invariants:
  - Stable mapping between vector UUIDs and FAISS ids (`_vector_uuid_to_faiss_int`).
  - Fusion output deterministic given identical inputs/config (`FusionConfig`, `RetrievalConfig`).
  - Chunk registry and dense store must stay in sync on add/remove operations.

## Configuration
- Env/config knobs (Pydantic models in `config.py`):
  - `HybridSearchConfig` – top-level settings (namespace mode, chunking, retrieval budgets).
  - `DenseIndexConfig` (TODO confirm env vars, e.g., `HYBRIDSEARCH_EXPECTED_NTOTAL`).
  - `FusionConfig` – RRF/MMR parameters (`k0`, `mmr_lambda`, `token_budget`, `byte_budget`).
  - `RetrievalConfig` – channel weights, top-k, filter policy.
- Configuration manager (`HybridSearchConfigManager`) loads JSON/YAML (TODO document sample path).
- Validate config: `direnv exec . python - <<'PY'` (TODO provide doctor command).

## Data contracts & schemas
- Typed dataclasses in `types.py` (e.g., `HybridSearchRequest`, `HybridSearchResponse`, `ValidationReport`).
- TODO: link JSON schema if exported (e.g., `docs/schemas/hybrid-search-request.json`).
- Chunk artifact formats: JSONL with `ChunkPayload` structure; vector dumps via `serialize_state`.

## Interactions with other packages
- Upstream: expects pre-generated embeddings/chunks (DocParsing pipeline) and optional lexical index implementations.
- Downstream: exposes ranked results and diagnostics consumed by application layer / API gateway.
- ID/path guarantees: vector IDs are UUIDs; chunk paths referenced by `ChunkRegistry`; FAISS snapshots stored alongside service state.

## Observability
- Logs: standard library logging via `Observability` (structured counters, histograms).
- Metrics/tracing: `MetricsCollector`, `TraceRecorder` emit samples that can be forwarded to Prometheus/OpenTelemetry (TODO detail exporters).
- **SLIs/SLOs**: TODO define success rate and latency targets; monitor `AdapterStats` for GPU availability.
- Health: TODO add/describe doctor or heartbeat command (e.g., `python -m DocsToKG.HybridSearch.service --healthcheck`).

## Security & data handling
- ASVS level: TODO (assume L2 while APIs exposed).
- Threats (STRIDE):
  - Spoofing: ensure query/auth handled by upstream gateway.
  - Tampering: protect FAISS snapshots and chunk artifacts (checksum, ACLs).
  - Repudiation: log search requests/namespace dispatch with correlation ids.
  - Information disclosure: sanitize highlights, avoid leaking embeddings; no PII stored by default.
  - DoS: enforce `FusionConfig.token_budget` and `RetrievalConfig.top_k` to cap workload.
- Data classification: no PII by default; embeddings treated as internal data; secrets (GPU credentials, index paths) managed via config.

## Development tasks
```bash
direnv exec . ruff check src/DocsToKG/HybridSearch tests/hybrid_search
direnv exec . mypy src/DocsToKG/HybridSearch
direnv exec . pytest tests/hybrid_search/test_suite.py -q
# TODO add fmt/lint/typecheck orchestrated via Justfile when available
```
- For GPU-specific tests, ensure FAISS w/ GPU available or skip markers (`pytest -q -k gpu --maxfail=1`).

## Agent guardrails
- Do:
  - Extend ingestion/search pipelines via protocols (`LexicalIndex`, `DenseVectorStore`).
  - Add metrics/observability using existing `Observability` hooks.
- Do not:
  - Change vector UUID to FAISS mapping without updating snapshot compatibility.
  - Bypass `ResultShaper` budgets or disable pagination verification.
- Danger zone:
  - `direnv exec . python -m DocsToKG.HybridSearch.store --rebuild-all` (TODO confirm command) may delete/rebuild FAISS indices.
  - Modifying serialization formats (`serialize_state`, `ChunkPayload`) without coordinated migrations.

## FAQ
- Q: How do I add a new dense store implementation?
  A: Implement `DenseVectorStore` protocol, wire into `FaissRouter` via factory, update `HybridSearchConfigManager`.

- Q: How can I run the end-to-end test suite?
  A: `direnv exec . pytest tests/hybrid_search/test_suite.py -q`; ensure optional GPU tests skipped or satisfied.

<!-- Machine-readable appendix -->
```json x-agent-map
{
  "entry_points":[
    {"type":"python","module":"DocsToKG.HybridSearch.service","symbols":["HybridSearchService","HybridSearchAPI"]},
    {"type":"python","module":"DocsToKG.HybridSearch.pipeline","symbols":["ChunkIngestionPipeline","Observability"]},
    {"type":"python","module":"DocsToKG.HybridSearch.store","symbols":["ManagedFaissAdapter","serialize_state","restore_state"]}
  ],
  "env":[
    {"name":"TODO_HYBRID_CONFIG_PATH","default":"configs/hybrid-search.yaml","required":false},
    {"name":"TODO_HYBRID_EXPECTED_NTOTAL","default":null,"required":false}
  ],
  "schemas":[
    {"kind":"python-typing","path":"src/DocsToKG/HybridSearch/types.py"}
  ],
  "artifacts_out":[
    {"path":"faiss_snapshots/*.bin","consumed_by":["service restore"]},
    {"path":"chunk_registry/*.jsonl","consumed_by":["lexical index loaders"]}
  ],
  "danger_zone":[
    {"command":"TODO_rebuild_command","effect":"Rebuilds or deletes FAISS indexes"}
  ]
}
```
