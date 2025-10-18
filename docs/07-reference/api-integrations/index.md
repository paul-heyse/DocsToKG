# 1. API Integrations

This section summarises the primary interfaces DocsToKG exposes and how to integrate
them with downstream systems.

## 2. Hybrid Search REST API

- **Endpoint**: `POST /v1/hybrid-search`
- **Implementation**: `DocsToKG.HybridSearch.service.HybridSearchAPI`
- **Payload**:
  - `query` (string, required)
  - `namespace` (string, optional)
  - `filters` (object, optional)
  - `page_size` (int, default 10)
  - `cursor` (string, optional pagination token)
  - `diversification` (bool) for MMR
  - `diagnostics` (bool) to include per-signal scores
- **Response**: Ranked list containing document metadata, highlight spans, fused score
  diagnostics, and pagination cursor.

Use the generated documentation in `docs/04-api/DocsToKG.HybridSearch.service.md` and `docs/04-api/DocsToKG.HybridSearch.types.md` for attribute-level detail.

## 3. Python Service Layer

`HybridSearchService` in `DocsToKG.HybridSearch.service` exposes a `search` method returning strongly-typed response models (`HybridSearchResponse`, `HybridSearchResult` defined in `types.py`).
Use this interface when embedding DocsToKG within other Python services.

```python
from DocsToKG.HybridSearch.types import HybridSearchRequest
from my_project.hybrid import build_hybrid_service  # assemble service per docs/06-operations/index.md

service = build_hybrid_service()
request = HybridSearchRequest(query="ontology alignment best practices", page_size=5)
response = service.search(request)
for hit in response.results:
    print(hit.doc_id, hit.score, hit.highlights)
```

## 4. CLI Utilities

- `python -m DocsToKG.ContentDownload.cli` – Batch download scholarly PDFs from Pyalex and other resolvers.
- `python -m DocsToKG.DocParsing.chunking` – Convert DocTags into chunked Markdown/metadata (see `--validate-only` and profile flags).
- `python -m DocsToKG.DocParsing.embedding` – Generate dense embeddings for chunked output.
- `python -m DocsToKG.OntologyDownload.cli pull` – Download and validate ontologies described in `sources.yaml`.

Each CLI supports `--help` for flags and is documented in `docs/06-operations/index.md`.

## 5. Event and Stream Hand-offs

DocsToKG does not yet ship a streaming ingestion layer. When integrating with external
systems:

1. Schedule ingestion jobs that populate the document registry and trigger chunk embedding.
2. Publish HybridSearch responses to your messaging infrastructure if latency budgets require asynchronous processing.
3. Use the observability utilities (`DocsToKG.HybridSearch.observability`) to expose metrics for downstream alerting.

See `docs/hybrid_search_runbook.md` for operational scenarios and rollback strategies.
