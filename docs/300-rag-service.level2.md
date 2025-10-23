# DocsToKG • RAG Service (Level-2 Spec)

## Purpose & Non-Goals
**Purpose:** Compose hybrid retrieval + KG expansion into **grounded answers** with provenance, budgets, and diagnostics.  
**Non-Goals:** Training LLMs or agent autonomy.

## HTTP API (Schemas)
**Request**
```json
{"query":"...","top_k":10,"namespace":"default","budget":{"max_chunks":48,"max_tokens_gen":0,"timeout_s":12},"kg_expansion":{"enabled":true,"hops":1,"limit":32},"diagnostics":true}
```
**Response**
```json
{"answer":"...","citations":[{"doc_id":"...","uuid":"...","page":3,"score":0.87,"snippet":"..."}],
 "diagnostics":{"timings_ms":{"dense":42,"lexical":29,"kg":11,"fuse":5,"synthesis":0,"total":96},"budget_used":{"chunks":32,"tokens_gen":0},"degraded":false}}
```

## Orchestration
Validate → Hybrid (RRF) → KG expansion → MMR diversify → budget apply → optional synthesis with strict grounding/citations.

## Config & Tuning
`evidence.max_snippet_chars=800`; strict guardrails; deterministic cache keys for no-gen.

## Observability & SLOs
Latency buckets for gen/no-gen; citation_coverage; degraded flags with reasons.

## Failure Modes
Missing snapshots → 503 degraded; timeout → partial evidence + degraded=true.

## Security
Redact snippets in logs; dev-mode reveal gated by flag.

## Tests
Planner, budget accounting, citation extraction, schema validation, retrieval→KG integration.
