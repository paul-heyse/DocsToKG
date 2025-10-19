# Project Context

## Purpose
DocsToKG is an end-to-end document-to-knowledge-graph pipeline that turns heterogeneous source material into search-ready, ontology-aware assets. The program is organised as four tightly integrated Python packages:

- **ContentDownload** resolves scholarly works (PDF, HTML, XML) via providers such as OpenAlex, Unpaywall, Crossref, and institutional repositories, while enforcing polite crawling, manifest bookkeeping, and resumable downloads.
- **DocParsing** converts raw documents into DocTags, chunked passages, and dense/sparse embeddings using Docling, vLLM-hosted Qwen models, and SPLADE, producing schemas that downstream systems can ingest deterministically.
- **OntologyDownload** plans, fetches, and validates third-party ontologies (e.g., HP, GO, ICD) with hardened networking, checksum verification, and validator fan-out so that hybrid retrieval respects controlled vocabularies.
- **HybridSearch** orchestrates GPU-accelerated FAISS stores plus lexical backends to deliver low-latency hybrid retrieval, namespace routing, fusion (RRF + MMR), observability, and snapshot management.

Together these stages ingest raw content, enrich it with ontological context, and expose a programmable hybrid search surface optimised for AI agent consumption.

## Tech Stack
- **Language & runtime**: Python 3.12+ (some tooling supports 3.10/3.11) with dataclasses, `typing`, and protocol-based interfaces.
- **GPU / numerical**: Custom `faiss-1.12.0` CUDA 12 wheel (cuVS-capable), NumPy, OpenBLAS, CUDA libraries (`libcudart.so.12`, `libcublas.so.12`), jemalloc.
- **Model serving**: vLLM for Qwen dense embeddings, Docling for DocTags extraction, SPLADE for sparse vectors.
- **CLI & orchestration**: `argparse`-driven CLIs in each package, composable manifest/telemetry subsystems, `ThreadPoolExecutor` for parallelism.
- **Observability & storage**: JSONL manifests, SQLite caches, structured logging (`logging_utils`, `telemetry`), optional OpenSearch-compatible lexical simulator.
- **Tooling**: `black`, `isort`, `ruff`, `mypy`, `pytest`, custom documentation validators, `direnv`, and `scripts/bootstrap_env.sh` for environment preparation.

## Project Conventions

### Code Style
- Follow repository-wide [STYLE_GUIDE.md](../docs/STYLE_GUIDE.md) and [CODE_ANNOTATION_STANDARDS.md](../docs/CODE_ANNOTATION_STANDARDS.md); NAVMAP headers are maintained at the top of modules to keep documentation and navigation tooling in sync.
- Enforce formatting with `black` and `isort`, lint with `ruff`, and type-check with `mypy`. Public APIs require comprehensive docstrings; examples must be executable.
- Keep docstrings and module-level READMEs authoritative—changes to behaviour must update both code and documentation before landing (documentation-first expectation).
- Avoid mutating configuration dataclasses directly; rely on helper constructors (`HybridSearchConfigManager`, `ResolvedConfig`, `settings.load_config`) to preserve immutability and validation invariants.

### Architecture Patterns
- **Stage separation**: Each package encapsulates a pipeline stage with clear inputs/outputs (ContentDownload manifests, DocParsing chunk/embedding JSONL, ontology lockfiles, FAISS snapshots).
- **Configuration via dataclasses**: `DenseIndexConfig`, `HybridSearchConfig`, `ResolverConfig`, and ontology `Settings` map 1:1 to CLI/environment overrides and govern runtime behaviour.
- **Manifest-driven idempotency**: Every stage writes JSONL manifests/SQLite indexes, computes hashes (SHA-256 by default), and resumes work deterministically across reruns.
- **GPU resource management**: HybridSearch leverages `faiss.StandardGpuResources`, cuVS toggles, FP16 configuration, and namespace-aware routing (`FaissRouter`) for multi-GPU scaling.
- **Plugin-ready design**: Ontology resolvers/validators and ContentDownload resolvers are discovered via entry points, enabling extensibility without modifying core orchestrators.
- **Observability baked-in**: Structured logging, telemetry sinks, and stats snapshots (HybridSearch `AdapterStats`, `Observability`) provide metrics for ingestion, search, and validation flows.

### Testing Strategy
- Prefer deterministic fixtures and manifests for all stages; HybridSearch ships regression suites (`tests/hybrid_search/test_suite.py`, `test_gpu_similarity.py`, `test_router.py`) covering FAISS GPU kernels, namespace routing, and snapshot round-tripping.
- ContentDownload/DocParsing rely on pytest suites with synthetic artifacts, verifying resume logic, schema adherence, and resolver planning.
- Lint, type-check, and test in CI: `ruff check`, `mypy`, and targeted `pytest` invocations per package; GPU-specific tests assume the custom CUDA 12 wheel is available.
- Documentation validation scripts (`docs/scripts/validate_docs.py`, `check_links.py`) run alongside code tests to guarantee README accuracy and style compliance.

### Git Workflow
- Create feature branches off `main`; follow documentation-first workflow—author or update READMEs and AGENTS guides before merging functional changes.
- Commit messages should be imperative and scoped; use PR templates to describe scope, testing, documentation updates, and link to relevant specs.
- Automated checks (formatting, linting, type-checking, tests, docs validation) must pass prior to review; reviewers expect manifests/configs to remain idempotent and reproducible.
- Use GitHub issues/discussions for planning; larger changes should include design docs or updates to `openspec/` specs before implementation.

## Domain Context
DocsToKG operates across the scholarly content lifecycle:

- **Acquisition**: ContentDownload fetches scholarly PDFs/HTML/XML via resolver pipelines, handles caching, resumable downloads, polite rate limiting, and logs manifests (`manifest.jsonl`, SQLite indexes) for auditing.
- **Parsing & enrichment**: DocParsing transforms documents into DocTags using Docling+vLLM, splits them into namespace-aware chunks, and generates dense (Qwen) plus sparse (SPLADE, BM25) embeddings stored as JSONL.
- **Ontology management**: OntologyDownload ensures controlled vocabularies are current and validated (ROBOT, rdflib, Arelle) so extracted entities align with curated ontologies.
- **Hybrid retrieval**: HybridSearch fuses lexical indices (BM25/SPLADE) and GPU-backed FAISS vectors, supports namespace routing, warm snapshots, Metrics/Observability, and API workflows for synchronous search consumers.

Domain invariants include deterministic chunk IDs, schema versioning, consistent doc IDs across stages, and carefully managed GPU resources to guarantee reproducible results suitable for knowledge graph ingestion and AI agent tooling.

## Important Constraints
- **GPU and library availability**: HybridSearch depends on the custom FAISS CUDA 12 wheel, CUDA runtimes, and OpenBLAS; DocParsing requires GPUs for DocTags and embeddings when performance matters. Agents must not mutate `.venv` packages without approval (see AGENTS guides).
- **Environment layout**: Data roots (`DOCSTOKG_DATA_ROOT`, `LOCAL_ONTOLOGY_DIR`) follow documented directory structures; manifests and snapshots must remain co-located with their indexes.
- **Idempotency & determinism**: Manifest-driven resumes rely on stable SHA-256 hashes and configuration; altering hash algorithms or config defaults demands migration planning and documentation updates.
- **Politeness & credentials**: Resolver configurations honour robots.txt, per-domain token buckets, and demand environment-set API keys (Unpaywall, BioPortal, etc.); misuse risks blacklisting.
- **Snapshot safety**: HybridSearch snapshots must be converted to CPU-compatible bytes before persistence (`serialize_state`); GPU indexes are not directly serialised to disk.
- **Documentation-first process**: New behaviour requires READMEs, AGENTS, and `openspec/` specs to be updated before or alongside code changes.

## External Dependencies
- **Content acquisition services**: OpenAlex, Unpaywall, Crossref, Core, Semantic Scholar, DOAJ, ArXiv, Institutional repositories, plus optional Wayback/OAI endpoints via resolver plugins.
- **Ontology sources & validators**: BioPortal, OBO Foundry, Europe PMC, XBRL regulators; validators include ROBOT, rdflib, Arelle, Schematron; checksum manifests via `checksums.py`.
- **Model & ML tooling**: vLLM (Qwen models), Docling (DocTags), SPLADE (sparse embedding), Qwen model repos cached under `DOCSTOKG_QWEN_DIR`.
- **Search infrastructure**: FAISS GPU runtime, optional OpenSearch or lexical simulators for hybrid retrieval testing.
- **System libraries**: CUDA 12 runtime, OpenBLAS, jemalloc, libgomp; `direnv` and `scripts/bootstrap_env.sh` orchestrate environments with pre-built wheels.
- **Observability & storage**: JSONL manifests, SQLite caches, structured logging pipelines; optional dashboards ingest logs/metrics produced by the telemetry subsystems.
