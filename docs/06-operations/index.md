# 1. Operations Guide

This guide covers day-two operations for DocsToKG, including ingestion jobs, search service management, ontology workflows, and scheduled maintenance.

## 2. Daily Workflow

1. **Monitor Dashboards** – Review hybrid search latency, self-hit accuracy, and ingestion throughput (see `docs/07-reference/monitoring/index.md`).
2. **Process Ingestion Queue** – Run content downloaders and DocParsing pipelines for new corpora.
3. **Validate Search Quality** – Execute the hybrid search validation harness and inspect diagnostics.
4. **Review Alerts** – Respond to documentation validation issues, ontology download failures, and hardware health warnings.

## 3. Running Core Pipelines

### 3.1 Content Acquisition

```bash
direnv exec . python -m DocsToKG.ContentDownload.cli \
  --topic "knowledge graphs" \
  --year-start 2022 --year-end 2024 \
  --out Data/PyalexRaw \
  --manifest Logs/manifest.jsonl \
  --workers 3 \
  --domain-min-interval example.org=0.75 \
  --staging
```

- Configure resolver credentials via environment variables (`UNPAYWALL_EMAIL`,
  `CORE_API_KEY`, `S2_API_KEY`, `DOAJ_API_KEY`) or `.envrc`.
- Use `--staging` to isolate each run under a timestamped directory. After the
  run, derive sidecars with the post-processing helpers:

  ```bash
  direnv exec . python tools/manifest_to_index.py runs/20250101_1200/manifest.jsonl \
    runs/20250101_1200/manifest.index.json
  direnv exec . python tools/manifest_to_csv.py runs/20250101_1200/manifest.jsonl \
    runs/20250101_1200/manifest.last.csv
  ```
- `--log-format csv` keeps JSONL manifests while adding a consolidated attempts
  CSV; invoke `tools/manifest_to_csv.py` to generate `manifest.last.csv` after
  the downloader finishes.
- Review `docs/content-download-migration.md` when upgrading existing
  automation to the new resolver stack and logging surfaces.

### 3.2 Document Parsing & Embedding

```bash
# Convert DocTags into chunked Markdown and metadata
direnv exec . docparse chunk \
  --in-dir Data/DocTagsFiles \
  --out-dir Data/ChunkedDocTagFiles \
  --workers 4 \
  --min-tokens 256 \
  --max-tokens 512

# Generate embeddings suitable for FAISS ingestion
direnv exec . docparse embed \
  --chunks-dir Data/ChunkedDocTagFiles \
  --out-dir Data/Embeddings \
  --profile gpu-default \
  --batch-size-qwen 96
```

- Adjust default input/output paths via CLI flags; see `docparse chunk --help` and `docparse embed --help` for environment defaults.
- Use the `--batch-size-qwen` option when processing large corpora on GPU hardware.

### 3.3 Hybrid Search Indexing

```python
import json
from pathlib import Path

from DocsToKG.HybridSearch.config import HybridSearchConfigManager
from DocsToKG.HybridSearch import Observability
from DocsToKG.HybridSearch.ingest import ChunkIngestionPipeline
from DocsToKG.HybridSearch.store import FaissVectorStore, ChunkRegistry, serialize_state
from DocsToKG.HybridSearch.storage import OpenSearchSimulator
from DocsToKG.HybridSearch.types import DocumentInput

config_manager = HybridSearchConfigManager(Path("config/hybrid_config.json"))
config = config_manager.get()
faiss = FaissVectorStore(dim=1536, config=config.dense)
registry = ChunkRegistry()
opensearch = OpenSearchSimulator()
pipeline = ChunkIngestionPipeline(
    faiss_index=faiss,
    opensearch=opensearch,
    registry=registry,
    observability=Observability(),
)

snapshot_path = Path("artifacts/faiss.snapshot.json")
docs_to_upsert = [
    DocumentInput(
        doc_id="whitepaper-001",
        namespace="public",
        chunk_path="Data/Embeddings/whitepaper-001.chunks.jsonl",
        vector_path="Data/Embeddings/whitepaper-001.vectors.jsonl",
        metadata={"source": "pyalex"},
    )
]
pipeline.upsert_documents(docs_to_upsert)
snapshot = serialize_state(faiss, registry)
snapshot_path.write_text(json.dumps(snapshot))
```

- This pipeline loads pre-computed chunk payloads, updates sparse stores, and rebuilds FAISS indexes when thresholds are met.
- Reuse the same registry instance between runs to support deletions and incremental updates.

### 3.4 Ontology Downloads

```bash
direnv exec . python -m DocsToKG.OntologyDownload.cli pull \
  --spec configs/sources.yaml \
  --force \
  --json > logs/ontology_pull.json
```

- Validate downloads with `direnv exec . python -m DocsToKG.OntologyDownload.cli validate <id>`.
- Maintain configuration under version control and rotate API credentials securely.
- Tune validation throughput with `defaults.validation.max_concurrent_validators`
  (1-8) and switch large ontologies to streaming normalization by lowering
  `defaults.validation.streaming_normalization_threshold_mb` when disk pressure
  is a concern.

## 4. Operating the Hybrid Search API

Integrate `HybridSearchAPI` into your preferred web framework. Example with FastAPI:

```python
from fastapi import FastAPI, HTTPException

from DocsToKG.HybridSearch import HybridSearchAPI
from my_project.hybrid import build_hybrid_service  # assemble service as shown above

app = FastAPI()
service = build_hybrid_service()
api = HybridSearchAPI(service)

@app.post("/v1/hybrid-search")
def hybrid_search(payload: dict):
    status, body = api.post_hybrid_search(payload)
    if status != 200:
        raise HTTPException(status_code=status, detail=body)
    return body
```

- Reload configuration dynamically using `HybridSearchConfigManager.reload`.
- Warm caches by issuing representative queries after startup.

## 5. Module Consolidation Checklist

- Review `docs/hybrid_search_module_migration.md` and update any automation that still imports from deprecated shims (`results`, `similarity`, `retrieval`, `schema`, `operations`, `tools`).
- Ensure CI workflows call `python -m DocsToKG.HybridSearch.validation` instead of legacy
  scripts.
- Watch for `DeprecationWarning` in logs and resolve them before upgrading to DocsToKG
  v0.6.0.

## 6. Scheduling & Automation

- **Cron / Airflow**: Schedule ingestion, embedding, and ontology jobs at off-peak hours.
- **CI/CD**: Automate documentation validation, link checks, and unit tests before deployments.
- **Maintenance Reminders**: Follow cadence outlined in `docs/DOCUMENTATION_MAINTENANCE_SCHEDULE.md`.

## 7. Backups & Recovery

1. Serialize FAISS and registry state (`serialize_state`) before schema changes.
2. Store snapshots and ontology manifests in durable object storage with ≥30-day retention.
3. Test restores quarterly using the runbook in `docs/hybrid_search_runbook.md`.
4. Document restore outcomes in an operational log for auditability.

## 8. Incident Response

- Consult `docs/hybrid_search_runbook.md` for failover plans and validation routines.
- Escalate blocking issues through the channels defined in `CONTRIBUTING.md`.
- Capture root-cause analyses and feed improvements back into this document.

Keeping this guide current is part of the monthly documentation review. Submit updates when procedures change or new tools are introduced.
