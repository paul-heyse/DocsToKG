# DocsToKG • RAG Service — Subsystem Architecture

## Purpose
Compose **grounded answers** by orchestrating HybridSearch + KnowledgeGraph with budgeted retrieval and explicit provenance.

## Responsibilities
- Request validation, query planning, channel fan-out, fusion, **grounding** (citation-first), optional LLM answer synthesis.
- Expose agent-friendly API responses (JSON) with structured citations.

## Request Model (v0)
```json
{
  "query": "string",
  "top_k": 8,
  "namespace": "default",
  "budget": { "max_chunks": 48, "max_tokens_gen": 1024, "timeout_s": 12 },
  "kg_expansion": { "enabled": true, "hops": 1, "limit": 32 }
}
```

## Orchestration Plan
1. Hybrid retrieval → RRF fuse.
2. KG expansion by concepts/triangles.
3. Dedup + MMR; apply budgets.
4. Grounded synthesis (optional); strict citation format.
5. Return structured answer + diagnostics.

## Response Model (v0)
```json
{
  "answer": "string (may be empty if synthesis disabled)",
  "citations": [
    {"doc_id":"...", "uuid":"...", "page":3, "score":0.87, "snippet":"..."}
  ],
  "diagnostics": {
    "timings_ms": {...},
    "channels": {"lexical": {...}, "dense": {...}},
    "fusion": {"method":"RRF+MMR", "k":8},
    "budget_used": {"chunks": 32, "tokens_gen": 512}
  }
}
```
