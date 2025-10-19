# 1. Technical Reference

This section contains subsystem deep-dives, integration guides, and operational runbooks that complement the architecture and setup documentation.

## 2. Reference Materials

- 🔗 **[FAISS Integration](./faiss/index.md)** – GPU/CPU deployment notes, wheel provenance, and index lifecycle management aligned with `DocsToKG.HybridSearch.store`.
- 📚 **[External Dependencies](./dependencies/index.md)** – Curated list of Python packages, CUDA wheels, and upstream services used across the repo (`DocsToKG.ContentDownload`, `DocParsing`, `HybridSearch`, `OntologyDownload`).
- 📖 **[Glossary](./glossary.md)** – Definitions covering terminology surfaced in manifests, telemetry events, and hybrid search scoring.

## 3. Integration Guides

- 🔌 **[API Integrations](./api-integrations/index.md)** – Examples for embedding `HybridSearchAPI` into FastAPI and wiring resolver webhooks.
- 🔧 **[Tool Configuration](./tooling/index.md)** – Bootstrap, linting, and documentation automation scripts (`docs/scripts/*`, `scripts/bootstrap_env.sh`, `openspec` CLI).

## 4. Advanced Topics

- 🚀 **[Performance Optimization](./performance/index.md)** – Chunking heuristics, FAISS tuning, and Docling configuration tweaks validated through the latest benchmarks.
- 🔒 **[Security Considerations](./security/index.md)** – Secrets management, artifact isolation, and supply-chain controls mapped to the current module layout.
- 🧪 **[Testing Strategies](./testing/index.md)** – Marker definitions, suite breakdowns, and CI recommendations for exercising optional GPU-backed flows.

## 5. Maintenance and Operations

- 🔄 **[Deployment Guide](./deployment/index.md)** – Environment bootstrap, data seeding, and release validation steps referencing the latest CLI entry points.
- 📊 **[Monitoring and Observability](./monitoring/index.md)** – Metrics, logging sinks, and dashboard recommendations for ingestion and hybrid search workloads.
- 🚨 **[Troubleshooting](./troubleshooting/index.md)** – Common failure modes (resolver throttling, ontology validation, FAISS rollbacks) with pointers to recovery commands.

Explore these references alongside `docs/06-operations/index.md` for day-two operations and `docs/05-development/index.md` for contributor workflows.
