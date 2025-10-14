# API Integrations

This section summarises the primary interfaces DocsToKG exposes and how to integrate them with downstream systems.

## Hybrid Search REST API

- **Endpoint**: `POST /v1/hybrid-search`
- **Implementation**: `DocsToKG.HybridSearch.api.HybridSearchAPI`
- **Payload**:
  - `query` (string, required)
  - `namespace` (string, optional)
  - `filters` (object, optional)
  - `page_size` (int, default 10)
  - `cursor` (string, optional pagination token)
  - `diversification` (bool) for MMR
  - `diagnostics` (bool) to include per-signal scores
- **Response**: Ranked list containing document metadata, highlight spans, fused score diagnostics, and pagination cursor.

Use the generated documentation in `docs/04-api/api.md` and `docs/04-api/retrieval.md` for attribute-level detail.

## Python Service Layer

`HybridSearchService` in `DocsToKG.HybridSearch.retrieval` exposes a `search` method returning strongly-typed response models (`HybridSearchResponse`, `HybridSearchResult`). Use this interface when embedding DocsToKG within other Python services.

```python
from DocsToKG.HybridSearch.retrieval import HybridSearchRequest
from my_project.hybrid import build_hybrid_service  # assemble service per docs/06-operations/index.md

service = build_hybrid_service()
request = HybridSearchRequest(query="ontology alignment best practices", page_size=5)
response = service.search(request)
for hit in response.results:
    print(hit.doc_id, hit.score, hit.highlights)
```

## CLI Utilities

- `python -m DocsToKG.ContentDownload.download_pyalex_pdfs` – Batch download scholarly PDFs from the Pyalex API.
- `python -m DocsToKG.DocParsing.run_docling_html_to_doctags_parallel` – Parallel HTML parsing into DocTags structures.
- `python -m DocsToKG.OntologyDownload.cli pull` – Download and validate ontologies described in `sources.yaml`.

Each CLI supports `--help` for flags and is documented in `docs/06-operations/index.md`.

## Event and Stream Hand-offs

DocsToKG does not yet ship a streaming ingestion layer. When integrating with external systems:

1. Schedule ingestion jobs that populate the document registry and trigger chunk embedding.
2. Publish HybridSearch responses to your messaging infrastructure if latency budgets require asynchronous processing.
3. Use the observability utilities (`DocsToKG.HybridSearch.observability`) to expose metrics for downstream alerting.

See `docs/hybrid_search_runbook.md` for operational scenarios and rollback strategies.
