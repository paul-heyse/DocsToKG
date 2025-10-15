# Deployment Guide

Use this reference when promoting DocsToKG from development to staging or production.

## Prerequisites

- Python 3.12 runtime with virtual environment support
- Access to object storage for FAISS snapshots and ontology artifacts
- Credentials for upstream data sources (Pyalex, ontology APIs)
- Observability stack (Prometheus/OpenTelemetry) for metrics ingestion

## Deployment Steps

1. **Prepare Environment**
   - Create isolated virtual environment.
   - Install project dependencies: `pip install -e .`.
   - Install GPU wheels as needed: `pip install -r requirements.in`.
2. **Seed Artifacts**
   - Download required ontologies: `python -m DocsToKG.OntologyDownload.cli pull`.
   - Build FAISS indexes using ingestion pipelines and `serialize_state`.
3. **Configure Runtime**
   - Populate `config/hybrid_config.json` (see `DocsToKG.HybridSearch.config` models).
   - Set environment variables: `HYBRID_SEARCH_CONFIG`, `DOCSTOKG_DATA_ROOT`, etc.
4. **Launch Services**
   - Start REST layer (FastAPI/uvicorn entrypoint if embedding into a service).
   - Schedule ingestion jobs (cron, Airflow, or manual invocations).
5. **Validate**
   - Run `pytest -m real_vectors --real-vectors`.
   - Execute `python -m DocsToKG.HybridSearch.validation`.
   - Perform smoke queries via REST API.

## Rollback

- Restore previous FAISS snapshot via `restore_state`.
- Revert ontology store to prior manifest.
- Deploy prior configuration commit (git tag or release).
- Use `docs/hybrid_search_runbook.md` for detailed rollback actions.

## Continuous Delivery Tips

- Store configuration and artifacts version numbers alongside releases.
- Automate documentation generation and attach outputs to release assets.
- Validate migrations (ontology schema changes, chunking tweaks) in staging before rolling forward.
