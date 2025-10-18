# 1. Deployment Guide

Use this guide when promoting DocsToKG from development to staging or production. It consolidates environment bootstrap steps, data seeding pipelines, and validation routines tested across the current codebase.

## 2. Pre-Deployment Checklist

- ✅ Source tree synced to the target ref (`git status` clean except for intentional changes).
- ✅ Secrets prepared: `PA_ALEX_KEY`, resolver API keys, ontology credentials, object-storage access keys.
- ✅ Target infrastructure sized for Docling chunking (CPU/RAM) and FAISS (GPU/CPU, OpenBLAS).
- ✅ Observability endpoints reachable (Prometheus/OpenTelemetry or log shipping agent).
- ✅ Change proposal or runbook reviewed (see `openspec/changes/…` for active specs).

## 3. Prepare the Runtime

```bash
./scripts/bootstrap_env.sh
direnv allow                      # loads .envrc for subsequent shells
direnv exec . python -m pip list  # sanity check packages from bundled wheels

direnv exec . python docs/scripts/validate_code_annotations.py
direnv exec . python docs/scripts/validate_docs.py
direnv exec . pytest -m smoke     # lightweight confidence suite
```

- `bootstrap_env.sh` installs Python 3.13 inside `.venv` and pulls the curated `requirements.txt`, including local wheels under `ci/wheels/`.
- `.envrc` wires `DOCSTOKG_DATA_ROOT`, `PYTHONPATH`, and `VIRTUAL_ENV` when accessed via `direnv exec . …`.
- Keep `.env` (or secret manager entries) with deployment-specific paths:
  ```env
  DOCSTOKG_DATA_ROOT=/srv/docstokg
  HYBRID_SEARCH_CONFIG=/etc/docstokg/hybrid_config.json
  ONTOLOGY_FETCHER_CONFIG=/etc/docstokg/sources.yaml
  ```

## 4. Seed Required Artifacts

### 4.1 Content Acquisition (OpenAlex + Resolvers)

```bash
direnv exec . python -m DocsToKG.ContentDownload.cli \
  --topic "knowledge graphs" \
  --year-start 2023 --year-end 2024 \
  --out $DOCSTOKG_DATA_ROOT/downloads \
  --manifest $DOCSTOKG_DATA_ROOT/manifests/manifest.jsonl \
  --workers 3 \
  --staging
```

- Resolver configuration lives in YAML (see `DocsToKG.ContentDownload.args.resolve_config`); pass it via `--resolver-config`.
- JSONL manifests and rotating sinks are emitted by `DocsToKG.ContentDownload.telemetry`. Summaries appear under `Logs/*.csv` when `LastAttemptCsvSink` is enabled.
- Use `docs/07-reference/tooling/index.md` for manifest-to-index conversion helpers.

### 4.2 DocParsing Chunking Pipeline

```bash
direnv exec . python -m DocsToKG.DocParsing.chunking \
  --in-dir $DOCSTOKG_DATA_ROOT/doctags \
  --out-dir $DOCSTOKG_DATA_ROOT/chunks \
  --min-tokens 256 \
  --max-tokens 512 \
  --workers 4
```

- This forwards into `DocsToKG.DocParsing._chunking.runtime.main`, so CLI flags match the module documented in `docs/06-operations/index.md`.
- Telemetry for chunking runs through `DocsToKG.DocParsing.telemetry.StageTelemetry`; ensure `$DOCSTOKG_DATA_ROOT/logs/chunking` is writable.

### 4.3 Embedding Pipeline

```bash
direnv exec . python -m DocsToKG.DocParsing.embedding \
  --chunks $DOCSTOKG_DATA_ROOT/chunks \
  --vectors $DOCSTOKG_DATA_ROOT/embeddings \
  --model sentence-transformers/all-mpnet-base-v2 \
  --batch-size 96
```

- Relies on `torch`, `xformers`, and `cupy` wheels installed during bootstrap; confirm CUDA 12.9 drivers for GPU use.
- Optional VLLM acceleration can be toggled via `--execution-backend=vllm` once the cluster has GPU capacity.

### 4.4 Hybrid Search Indexing

```python
from pathlib import Path

from DocsToKG.HybridSearch.config import HybridSearchConfigManager
from DocsToKG.HybridSearch.ingest import ChunkIngestionPipeline
from DocsToKG.HybridSearch.observability import Observability
from DocsToKG.HybridSearch.storage import ChunkRegistry, OpenSearchSimulator
from DocsToKG.HybridSearch.types import DocumentInput
from DocsToKG.HybridSearch.vectorstore import FaissVectorStore, serialize_state

config = HybridSearchConfigManager(Path("/etc/docstokg/hybrid_config.json")).get()
faiss = FaissVectorStore(dim=config.dense.dim, config=config.dense)
registry = ChunkRegistry()
opensearch = OpenSearchSimulator()
pipeline = ChunkIngestionPipeline(
    faiss_index=faiss,
    opensearch=opensearch,
    registry=registry,
    observability=Observability(),
)
pipeline.upsert_documents([
    DocumentInput(
        doc_id="whitepaper-001",
        namespace="public",
        chunk_path=f"{DOCSTOKG_DATA_ROOT}/chunks/whitepaper-001.jsonl",
        vector_path=f"{DOCSTOKG_DATA_ROOT}/embeddings/whitepaper-001.jsonl",
        metadata={"source": "pyalex"},
    )
])
snapshot = serialize_state(faiss, registry)
Path("/srv/docstokg/faiss.snapshot.json").write_text(snapshot.model_dump_json())
```

- Promote the FAISS snapshot (and registry manifest) to durable object storage after each ingestion.
- For production OpenSearch/Elastic backends replace `OpenSearchSimulator` with the real adapter from `DocsToKG.HybridSearch.storage`.

### 4.5 Ontology Downloads

```bash
direnv exec . python -m DocsToKG.OntologyDownload.cli pull \
  --spec /etc/docstokg/sources.yaml \
  --force \
  --json > $DOCSTOKG_DATA_ROOT/logs/ontology_pull.json

direnv exec . python -m DocsToKG.OntologyDownload.cli validate hp latest
```

- Settings classes in `DocsToKG.OntologyDownload.settings` expect `pydantic-settings`; ensure configuration files reside alongside environment entries referenced in the docstring.
- Run `cli plan` for dry-run previews and `cli prune` to remove stale artifacts between releases.

## 5. Configure and Launch Services

1. **Configuration Management**
   - Maintain `hybrid_config.json` with `HybridSearchConfigManager`, keeping dense/sparse weights in sync with evaluation.
   - Store ontologies under `$DOCSTOKG_DATA_ROOT/ontologies`—the downloader writes manifests referencing version hashes and storage locations.

2. **Service Startup**
   ```bash
   direnv exec . uvicorn myproject.app:app --host 0.0.0.0 --port 8080
   ```
   - The `FastAPI` example in `docs/06-operations/index.md` wraps `HybridSearchAPI`.
   - Warm caches by issuing representative search requests (`DocsToKG.HybridSearch.validation.run_smoke_suite`).

3. **Scheduled Jobs**
   - Use Airflow/cron/Kubernetes Jobs to invoke:
     - `python -m DocsToKG.ContentDownload.cli …`
     - `python -m DocsToKG.DocParsing.chunking …`
     - `python -m DocsToKG.DocParsing.embedding …`
     - `python -m DocsToKG.OntologyDownload.cli pull …`

## 6. Validation Before Flip

- `direnv exec . python -m DocsToKG.HybridSearch.validation` (runs fusion, SPLADE, and dense recall checks).
- `direnv exec . pytest -m "hybrid_search or ontology"` to exercise storage, telemetry, and CLI shims.
- `direnv exec . python docs/scripts/validate_code_annotations.py` and `docs/scripts/validate_docs.py` keep auto-generated documentation in sync.
- Issue smoke searches against `/v1/hybrid-search` (expect 200/JSON payload). Confirm telemetry shipping via Prometheus exporter if enabled (`prometheus-fastapi-instrumentator`).

## 7. Rollback and Recovery

- **FAISS Snapshot**: Rehydrate the previous snapshot with `DocsToKG.HybridSearch.vectorstore.restore_state`.
- **Ontology Artifacts**: Re-run `cli prune --dry-run` to identify erroneous downloads, then redeploy the prior manifest version.
- **Service Config**: Revert `hybrid_config.json` in configuration management and restart the API pods.
- See `docs/hybrid_search_runbook.md` for a deeper incident response checklist (cache warming, feature flag rollbacks, data root restoration).

## 8. Continuous Delivery Tips

- Version control configuration (`configs/`, `hybrid_config.json`) alongside application releases; tag artifacts with the same release identifier stored in manifests.
- Automate `direnv exec . scripts/run_precommit.sh`, documentation validation, and `pytest -m smoke` in CI before promoting builds.
- Stage ontology schema migrations and chunking parameter changes in a pre-production environment, then capture telemetry deltas for review.
