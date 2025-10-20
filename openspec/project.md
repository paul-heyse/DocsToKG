# Project Context

## Purpose
DocsToKG is an end-to-end document-to-knowledge-graph pipeline that turns heterogeneous source material into search-ready, ontology-aware assets. The program is organised as four tightly integrated Python packages:

- **ContentDownload** resolves scholarly works (PDF, HTML, XML) via providers such as OpenAlex, Unpaywall, Crossref, and institutional repositories, while enforcing polite crawling, manifest bookkeeping, and resumable downloads.
- **DocParsing** converts raw documents into DocTags, chunked passages, and dense/sparse embeddings using Docling, vLLM-hosted Qwen models, and SPLADE, producing schemas that downstream systems can ingest deterministically.
- **OntologyDownload** plans, fetches, and validates third-party ontologies (e.g., HP, GO, ICD) with hardened networking, checksum verification, and validator fan-out so that hybrid retrieval respects controlled vocabularies.
- **HybridSearch** orchestrates GPU-accelerated FAISS stores plus lexical backends to deliver low-latency hybrid retrieval, namespace routing, fusion (RRF + MMR), observability, and snapshot management.

Together these stages ingest raw content, enrich it with ontological context, and expose a programmable hybrid search surface optimised for AI agent consumption.

## Tech Stack
- **Language & runtime**: Python 3.12+ (tooling remains compatible with 3.10/3.11 during transitional periods) using dataclasses, protocols, and type hints for explicit contracts.
- **GPU / numerical**: Custom `faiss-1.12.0` CUDA 12 wheel with cuVS support, NumPy/OpenBLAS, CUDA runtime libraries (`libcudart.so.12`, `libcublas.so.12`), jemalloc, libgomp.
- **Model serving**: vLLM for Qwen dense embeddings, Docling + Granite DocTags models, SPLADE for sparse representations; model caches managed via `DOCSTOKG_*` environment variables.
- **CLI & orchestration**: Package-specific `argparse` CLIs, `ThreadPoolExecutor`-based concurrency, manifest and telemetry subsystems for consistent process control.
- **Observability & storage**: Structured logging (JSONL + console), SQLite manifest indices, FAISS snapshots, namespace stats; optional OpenSearch simulator for lexical testing.
- **Tooling & automation**: `black`, `isort`, `ruff`, `mypy`, `pytest`, `direnv`, `scripts/bootstrap_env.sh`, documentation validators (`validate_docs.py`, `check_links.py`), and CI workflows enforcing documentation-first commits.

## Project Conventions

### Code Style
- Follow repository-wide [STYLE_GUIDE.md](../docs/STYLE_GUIDE.md) and [CODE_ANNOTATION_STANDARDS.md](../docs/CODE_ANNOTATION_STANDARDS.md); NAVMAP headers are maintained at the top of modules to keep documentation and navigation tooling in sync.
- Enforce formatting with `black` and `isort`, lint with `ruff`, and type-check with `mypy`. Public APIs require comprehensive docstrings; examples must be executable.
- Keep docstrings and module-level READMEs authoritative—changes to behaviour must update both code and documentation before landing (documentation-first expectation).
- Avoid mutating configuration dataclasses directly; rely on helper constructors (`HybridSearchConfigManager`, `ResolvedConfig`, `settings.load_config`) to preserve immutability and validation invariants.

### Architecture Patterns
- **Stage separation**: Four primary packages encapsulate ingestion, parsing, ontology management, and retrieval. Each stage consumes the previous stage’s outputs (manifests, chunk/embedding JSONL, ontologies, FAISS snapshots) and maintains clearly defined schemas.
- **Configuration via dataclasses**: `DenseIndexConfig`, `HybridSearchConfig`, `ResolverConfig`, ontology `Settings`, and DocParsing configuration helpers mirror CLI/environment overrides and enforce validation before execution.
- **Manifest-driven idempotency**: Stages write append-only JSONL manifests plus SQLite indexes, compute SHA-256 hashes, and respect resume logic to avoid duplicated work.
- **GPU resource management**: HybridSearch uses `faiss.StandardGpuResources`, cuVS detection, FP16 options, replication/sharding, and namespace routers for multi-GPU deployments; DocParsing embeddings rely on GPU-backed vLLM servers.
- **Plugin-ready design**: ContentDownload and OntologyDownload use entry-point registries for resolvers/validators, enabling external extensions while preserving core invariants.
- **Observability baked-in**: Structured logs, telemetry counters/histograms, `AdapterStats`, `Observability`, and diagnostic commands provide visibility across ingestion, validation, and search flows. Documentation and AGENTS guides include canonical metrics/expectations.
- **Documentation-first**: Every module includes NAVMAP headers, READMEs, and AGENTS guides; behaviour changes must accompany documentation updates and spec revisions.

### Testing Strategy
- **Unit & integration coverage**:
  - HybridSearch: `tests/hybrid_search/test_suite.py` (end-to-end ingestion/search), `test_gpu_similarity.py`, `test_router.py`, `test_store_snapshot.py`, `test_validator_resources.py` to guard FAISS GPU operations, namespace routing, snapshots, and validator memory budgets.
  - DocParsing: tests for chunk/embedding manifests, resume logic, hashing defaults, CLI planners (`plan`, `manifest`, `token-profiles`), and schema validation.
  - ContentDownload: resolver planning, manifest writing, resume caches, rate limiting, networking resilience, storage layout.
  - OntologyDownload: resolver planning/diffing, streaming downloads, validator execution, checksum enforcement, lockfile generation.
- **Tooling**: `ruff check`, `black --check`, `isort --check`, `mypy`, and targeted `pytest` commands per package; GPU-dependent tests assume the FAISS CUDA 12 wheel and CUDA libraries are installed.
- **Documentation quality**: `docs/scripts/validate_docs.py`, `check_links.py`, `generate_all_docs.py` ensure READMEs/AGENTS/specs remain accurate and style-compliant.
- **Performance verification**: Optional scale tests (`test_suite.py::test_hybrid_scale_suite`) monitor latency/throughput budgets; ingestion/resolver benchmarks recorded in manifests or summary logs.

### Git Workflow
- Create feature branches off `main`; follow documentation-first workflow—author or update READMEs and AGENTS guides before merging functional changes.
- Commit messages should be imperative and scoped; use PR templates to describe scope, testing, documentation updates, and link to relevant specs.
- Automated checks (formatting, linting, type-checking, tests, docs validation) must pass prior to review; reviewers expect manifests/configs to remain idempotent and reproducible.
- Use GitHub issues/discussions for planning; larger changes should include design docs or updates to `openspec/` specs before implementation.

## Domain Context
- **Content acquisition (ContentDownload)**: Resolver pipelines (OpenAlex, Unpaywall, Crossref, Core, DOAJ, etc.) fetch scholarly artifacts, honour robots.txt and centralized rate limiter policies, and persist manifests (`manifest.jsonl`, `manifest.sqlite3`, summary JSON) for auditing/resume. Storage layout includes staging directories, SHA-256 digests, and content-addressed options.
- **Document parsing & enrichment (DocParsing)**: DocTags conversion (Docling + Granite models), chunking heuristics, embedding generation (Qwen/vLLM, SPLADE), and manifest telemetry produce chunk/embedding JSONL with consistent UUIDs and hash-based idempotency. Schema validation ensures compatibility with HybridSearch ingestion.
- **Ontology management (OntologyDownload)**: Planning and fetching ontologies with resolver catalogs, rate limiting, TLS enforcement, checksum verification, and validator pipelines (ROBOT, rdflib, Arelle, schematron). Outputs include versioned ontology directories, lockfiles, manifests, and validator reports.
- **Hybrid retrieval (HybridSearch)**: GPU-accelerated FAISS ingestion, namespace routing (`FaissRouter`), snapshot/restore, synchronous search APIs with RRF/MMR fusion, observability (latency histograms, AdapterStats), and ingestion interoperability with DocParsing outputs.
- **Cross-stage invariants**: Deterministic hashing (default SHA-256), manifest-driven resumability, doc/chunk ID consistency, namespace preservation, and GPU resource coordination ensure reproducible knowledge graph ingestion and agent-ready retrieval.

- **GPU and library availability**: HybridSearch requires the FAISS CUDA 12 wheel plus CUDA/OpenBLAS/Jemalloc; DocParsing embeddings and DocTags benefit from GPUs/vLLM. Agents are forbidden from altering `.venv` without approval (mandated in AGENTS guides).
- **Environment layout**: Data roots (`DOCSTOKG_DATA_ROOT`, `LOCAL_ONTOLOGY_DIR`, `DOCSTOKG_MODEL_ROOT`) must follow documented directory structures; manifests, snapshots, and caches rely on colocation for resumability.
- **Idempotency & determinism**: Manifest-based resumes depend on stable SHA-256 hashes, consistent configuration, and append-only logging. Changing hash algorithms (`DOCSTOKG_HASH_ALG`) or defaults requires migration planning and documentation updates.
- **Politeness & credentials**: Resolvers obey robots.txt, centralized limiter policies, global dedupe, and require environment credentials (Unpaywall email, BioPortal API keys). Violations risk throttling or bans.
- **Snapshot safety**: HybridSearch snapshots must be serialised via CPU conversions (`serialize_state`/`restore_state`) with metadata; direct GPU serialization is unsupported.
- **Documentation-first**: Behavioural changes demand synchronized updates to module READMEs, AGENTS guides, and relevant specs (`openspec/`). NAVMAP headers must stay accurate.
- **Security & compliance**: Ontology downloads enforce TLS, checksum validation, controlled extraction (zip traversal protection), and maintain audit logs; content downloads respect license constraints and resolver policies.
- **Agent guardrails**: AGENTS guides dictate no-install workflows, canonical commands, and troubleshooting; agents must adhere to guardrails to prevent environment drift or policy violations.

- **Content acquisition services**: OpenAlex, Unpaywall, Crossref, Core, Semantic Scholar, DOAJ, arXiv, Europe PMC, institutional repositories, Wayback, OAI-PMH endpoints; resolvers may require API keys/headers.
- **Ontology sources & validators**: BioPortal, OBO Foundry, Arelle, ROBOT, rdflib, schematron bundles, checksum manifests; external archives (Zenodo, Figshare) supporting ontology releases.
- **Model & ML tooling**: vLLM (Qwen embedding models), Docling + Granite DocTags models, SPLADE, tokenizer assets; caches controlled via `DOCSTOKG_QWEN_DIR`, `DOCSTOKG_SPLADE_DIR`, `DOCSTOKG_MODEL_ROOT`.
- **Search infrastructure**: FAISS GPU runtime, optional OpenSearch simulators, CUDA 12 libraries, OpenBLAS, jemalloc, libgomp, `faiss.should_use_cuvs` heuristics.
- **Tooling & automation**: `direnv`, `scripts/bootstrap_env.sh`, GitHub Actions workflows, documentation scripts, manifest summarizers, telemetry consumers.
- **Observability & storage**: JSONL manifests/logs, SQLite caches, snapshot directories, log rotations; dashboards ingest logs for latency/error monitoring.
