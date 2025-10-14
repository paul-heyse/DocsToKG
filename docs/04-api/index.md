# API Reference

DocsToKG exposes a single HTTP-style interface for hybrid search through the `HybridSearchAPI` class in `DocsToKG.HybridSearch.api`. This reference describes the request/response schema and shows how to integrate the API into a web service.

> ℹ️  DocsToKG does not ship a standalone web server. Use the snippets below to wrap `HybridSearchAPI` with your preferred framework (FastAPI, Flask, etc.).

## Base Endpoint

```
POST /v1/hybrid-search
Content-Type: application/json
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Natural language search query |
| `namespace` | string | ❌ | Restrict search to a logical namespace |
| `filters` | object | ❌ | Additional metadata filters (values can be scalars or lists) |
| `page_size` | integer | ❌ | Number of results to return (default `10`) |
| `cursor` | string | ❌ | Pagination cursor returned from a previous call |
| `diversification` | boolean | ❌ | Enable Maximal Marginal Relevance (default `false`) |
| `diagnostics` | boolean | ❌ | Include per-signal scores (default `true`) |

Example request:

```json
{
  "query": "ontology alignment best practices",
  "namespace": "public",
  "page_size": 5,
  "filters": {
    "source": ["pyalex"],
    "year": "2024"
  },
  "diversification": true
}
```

### Response Body

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Ranked hits, highest score first |
| `next_cursor` | string \| null | Cursor for the next page (`null` when exhausted) |
| `total_candidates` | integer | Number of candidates evaluated across retrieval stages |
| `timings_ms` | object | Per-stage latency diagnostics |

Each result contains:

```json
{
  "doc_id": "whitepaper-001",
  "chunk_id": "5",
  "namespace": "public",
  "score": 23.41,
  "fused_rank": 1,
  "text": "Hybrid search combines lexical and dense retrieval strategies...",
  "highlights": ["hybrid", "retrieval"],
  "provenance_offsets": [[128, 196]],
  "metadata": {
    "title": "Hybrid Search Strategies",
    "authors": ["Doe, Jane"],
    "year": 2024
  },
  "diagnostics": {
    "bm25": 12.34,
    "splade": 7.85,
    "dense": 3.22
  }
}
```

### Error Responses

| Status | Body | When it occurs |
|--------|------|----------------|
| `400 Bad Request` | `{"error": "Invalid payload"}` | Missing `query`, wrong types, or validation error |
| `500 Internal Server Error` | `{"error": "message"}` | Unexpected exception while processing the request |

Validation failures raise `RequestValidationError` inside the service; the API layer converts them into `400` responses.

## Pagination

The API returns at most `page_size` results per call. When additional results are available, `next_cursor` is populated. Supply this value in the next request to continue pagination.

```json
{
  "query": "hybrid retrieval",
  "cursor": "opaque-token-from-previous-response"
}
```

Cursor chains are stable across requests unless the underlying index changes significantly. Use `DocsToKG.HybridSearch.operations.verify_pagination` during integration tests to ensure continuity.

## Diagnostics

`timings_ms` contains latency measurements for each stage (BM25, SPLADE, FAISS, fusion) when observability is enabled. Per-result `diagnostics` expose individual signal scores, useful for ranking audits.

## Integrating with FastAPI

```python
from pathlib import Path

from fastapi import FastAPI, HTTPException

from DocsToKG.HybridSearch.api import HybridSearchAPI
from my_project.hybrid import build_hybrid_service  # see docs/06-operations/index.md

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

Run locally with:

```bash
uvicorn app:app --reload --port 8000
```

Then issue a request:

```bash
curl -X POST http://localhost:8000/v1/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{"query": "knowledge graph embeddings", "page_size": 5}'
```

## Local Service Usage (No HTTP)

When embedding DocsToKG directly inside Python code, call the service layer once you have constructed it (see `docs/06-operations/index.md` for a full build walkthrough):

```python
from DocsToKG.HybridSearch.retrieval import HybridSearchRequest

service = build_hybrid_service()
request = HybridSearchRequest(query="ontology mapping pipeline")
response = service.search(request)
```

Working examples of service assembly live in `tests/conftest.py` and `tests/test_hybrid_search.py`.

## Related Resources

- `docs/06-operations/index.md` – Day-two operations and maintenance routines.
- `docs/07-reference/api-integrations/index.md` – Integration best practices and CLI helpers.
- `tests/test_hybrid_search.py` – Example assertions covering the service layer.
