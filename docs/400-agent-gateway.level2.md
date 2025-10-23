# DocsToKG • Agent Gateway (Level-2 Spec)

## Purpose & Non-Goals
**Purpose:** A constrained, auditable surface for autonomous agents to access RAG and lookups with quotas, RBAC, and deterministic defaults.  
**Non-Goals:** Free-form tools or external egress.

## API
- `POST /agents/v1/query` → wraps `/rag/query` with stricter budgets (default `max_chunks=24`, `max_tokens_gen=0`).
- `GET /agents/v1/concepts?text=...&top_k=...`
- `GET /agents/v1/status`

## AuthN/Z & Quotas
API keys per namespace; optional JWT. Roles: `READER`, `POWER`, `ADMIN`. Quotas: req/min, tokens/day, concurrent caps.

## Observability
`gateway_requests_total{route,role,code}`, `gateway_budget_denials_total{reason}`; audit log per call.

## Failure Modes
Budget exceeded → 429; downstream degraded → 503 pass-through.

## Security
Strict CORS; egress only to internal services; no raw vectors/snippets leakage.

## Tests
Quota/budget enforcer, RBAC, degraded mode propagation.
