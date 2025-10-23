# DocsToKG • Agent Gateway — Subsystem Architecture

## Purpose
Provide a **constrained, auditable** surface for autonomous agents to query/search/inspect with quotas and guardrails.

## Capabilities (v0)
- `agents.query_rag` → RAG request model (see RAG doc)
- `agents.lookup_concept` → by CURIE/label/synonym (top-k + metadata)
- `agents.get_document` → safe snippet fetch by `doc_id + page + span` (rate-limited)
- `agents.status` → snapshot age, corpus stats

## Guardrails
- **Quotas**: per-key requests/min, tokens/day (gen), chunk budget/request.
- **Allowed domains**: Gateway only; no agent-led scraping.
- **Determinism**: default non-gen path; generation allowed only with `allow_gen=true`.
- **Explainability**: responses include diagnostics and budget use.

## API Surface (HTTP+JSON)
- `POST /agents/v1/query` (wraps `/rag/query`)
- `GET /agents/v1/concepts?text=...&top_k=...`
- `GET /agents/v1/status`

## AuthN/Z
- API keys scoped to namespaces; optional JWT with org/project claims.
- RBAC: `READER`, `POWER`, `ADMIN`.

## Observability
- Per-key rate/latency/error histograms; budget denials; audit log for every call.
