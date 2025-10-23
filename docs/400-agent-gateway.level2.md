# DocsToKG • Agent Gateway (Level-2 Spec)

## Purpose & Non-Goals

**Purpose:** Provide a constrained, auditable API surface for autonomous agents to access RAG queries, concept lookups, and system status with strict quotas, RBAC enforcement, and comprehensive audit logging for safe agent-knowledge integration.

**Scope:**

- API contracts (REST/JSON endpoints for agents)
- RBAC model (roles, permissions, namespace scoping)
- Quota enforcement (rate limits, token budgets, concurrent caps)
- Audit logging (request tracking, security events)
- Request validation & sanitization
- Degraded mode propagation from downstream services

**Non-Goals:**

- Free-form tool execution (agents get read-only access)
- External egress (gateway only accesses internal DocsToKG services)
- Agent planning/orchestration (agents use gateway as knowledge source)
- Long-term conversation state (stateless request/response)

---

## API Contracts

### Endpoint: `POST /agents/v1/query`

**Purpose:** RAG query with stricter defaults for agent safety.

**Request:**

```json
{
  "query": "What are the phenotypic abnormalities associated with HP:0001250?",
  "namespace": "biomedical",
  "role": "READER",
  "budget": {
    "max_chunks": 24,
    "max_tokens_gen": 0,
    "timeout_s": 8
  },
  "kg_expansion": {
    "enabled": true,
    "hops": 1,
    "limit": 16
  },
  "allow_gen": false,
  "trace_id": "agent-query-12345"
}
```

**Response:**

```json
{
  "answer": "",
  "citations": [
    {
      "doc_id": "PMC8765432",
      "chunk_id": "PMC8765432#0042",
      "score": 0.87,
      "snippet": "Patients with seizures (HP:0001250)...",
      "rank": 1
    }
  ],
  "diagnostics": {
    "timings_ms": {"total": 156},
    "budget_used": {"chunks": 18, "tokens_gen": 0},
    "degraded": false
  },
  "quota_remaining": {
    "requests_per_minute": 42,
    "tokens_per_day": 450000
  }
}
```

**Differences from `/rag/query`:**

- Stricter defaults: `max_chunks=24` (vs 48), `max_tokens_gen=0` (vs 1024)
- `allow_gen` flag required for LLM synthesis
- Quota remaining included in response
- Trace ID for audit logging

---

### Endpoint: `GET /agents/v1/concepts`

**Purpose:** Lookup ontology concepts by text, CURIE, or synonym.

**Request:**

```
GET /agents/v1/concepts?text=seizures&top_k=5&ontologies=hp,mondo&trace_id=agent-lookup-789
```

**Response:**

```json
{
  "concepts": [
    {
      "curie": "HP:0001250",
      "iri": "http://purl.obolibrary.org/obo/HP_0001250",
      "label": "Seizure",
      "ontology": "hp",
      "ontology_version": "2025-09-15",
      "synonyms": ["Seizures", "Epileptic seizure"],
      "definition": "A seizure is an intermittent abnormality of nervous system...",
      "score": 1.0,
      "match_type": "exact"
    },
    {
      "curie": "MONDO:0005027",
      "label": "Epilepsy",
      "ontology": "mondo",
      "score": 0.85,
      "match_type": "related"
    }
  ],
  "quota_remaining": {
    "requests_per_minute": 48
  }
}
```

---

### Endpoint: `GET /agents/v1/status`

**Purpose:** System health and corpus statistics (non-sensitive metadata).

**Request:**

```
GET /agents/v1/status?namespace=biomedical&trace_id=agent-status-456
```

**Response:**

```json
{
  "status": "healthy",
  "namespace": "biomedical",
  "corpus_stats": {
    "documents": 150420,
    "chunks": 12054321,
    "concepts": 87543,
    "last_updated": "2025-10-23T00:00:00Z"
  },
  "services": {
    "hybrid_search": {"status": "healthy", "degraded": false},
    "knowledge_graph": {"status": "healthy", "degraded": false},
    "rag_service": {"status": "healthy", "degraded": false}
  },
  "snapshot_age_seconds": 312
}
```

---

## RBAC Model

### Roles & Permissions

| Role | Query RAG | Lookup Concepts | Get Status | Allow Gen | Max Chunks | Tokens/Day |
|------|-----------|-----------------|------------|-----------|------------|------------|
| `READER` | ✅ | ✅ | ✅ | ❌ | 24 | 0 |
| `POWER` | ✅ | ✅ | ✅ | ✅ | 48 | 100k |
| `ADMIN` | ✅ | ✅ | ✅ | ✅ | 100 | 500k |

### Permission Checks

```python
class PermissionChecker:
    def check_query_permission(self, role: Role, request: QueryRequest) -> PermissionResult:
        """Check if role can execute query with given parameters."""
        # Check generation permission
        if request.allow_gen and role not in [Role.POWER, Role.ADMIN]:
            return PermissionResult(
                allowed=False,
                reason="Generation requires POWER or ADMIN role"
            )
        
        # Check chunk budget
        if request.budget.max_chunks > role.max_chunks:
            return PermissionResult(
                allowed=False,
                reason=f"max_chunks {request.budget.max_chunks} exceeds role limit {role.max_chunks}"
            )
        
        # Check tokens budget
        if request.budget.max_tokens_gen > role.max_tokens_per_request:
            return PermissionResult(
                allowed=False,
                reason=f"max_tokens_gen {request.budget.max_tokens_gen} exceeds role limit {role.max_tokens_per_request}"
            )
        
        return PermissionResult(allowed=True)
```

### Namespace Scoping

```python
class NamespacePolicy:
    def __init__(self, api_key_namespaces: Dict[str, List[str]]):
        self.api_key_namespaces = api_key_namespaces
    
    def check_namespace_access(self, api_key: str, namespace: str) -> bool:
        """Check if API key has access to namespace."""
        allowed_namespaces = self.api_key_namespaces.get(api_key, [])
        return namespace in allowed_namespaces or "*" in allowed_namespaces
```

---

## Quota Enforcement

### Quota Types

| Quota Type | Scope | Default (READER) | Default (POWER) | Enforcement |
|------------|-------|------------------|-----------------|-------------|
| `requests_per_minute` | API key | 50 | 200 | Sliding window, 429 on exceed |
| `tokens_per_day` | API key | 0 | 100k | Daily reset, 429 on exceed |
| `max_concurrent` | API key | 5 | 20 | Active request count, 429 on exceed |
| `max_chunks_per_request` | Role | 24 | 48 | Request validation, 400 on exceed |

### Quota Tracker

```python
class QuotaTracker:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    def check_rate_limit(self, api_key: str, limit: int, window_seconds: int = 60) -> QuotaResult:
        """Check sliding window rate limit."""
        key = f"quota:rate:{api_key}"
        now = time.time()
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, now - window_seconds)
        
        # Count recent requests
        count = self.redis.zcard(key)
        
        if count >= limit:
            return QuotaResult(
                allowed=False,
                reason=f"Rate limit exceeded: {count}/{limit} requests per {window_seconds}s",
                remaining=0,
                reset_at=now + window_seconds,
            )
        
        # Add current request
        self.redis.zadd(key, {str(now): now})
        self.redis.expire(key, window_seconds)
        
        return QuotaResult(
            allowed=True,
            remaining=limit - count - 1,
            reset_at=now + window_seconds,
        )
    
    def check_token_quota(self, api_key: str, tokens_used: int, daily_limit: int) -> QuotaResult:
        """Check daily token quota."""
        key = f"quota:tokens:{api_key}:{date.today().isoformat()}"
        
        # Increment token count
        total_tokens = self.redis.incrby(key, tokens_used)
        self.redis.expire(key, 86400)  # Expire at end of day
        
        if total_tokens > daily_limit:
            return QuotaResult(
                allowed=False,
                reason=f"Daily token quota exceeded: {total_tokens}/{daily_limit}",
                remaining=0,
            )
        
        return QuotaResult(
            allowed=True,
            remaining=daily_limit - total_tokens,
        )
```

---

## Audit Logging

### Audit Log Schema

```json
{
  "timestamp": "2025-10-23T00:00:00Z",
  "trace_id": "agent-query-12345",
  "api_key_hash": "sha256:abc...",
  "role": "READER",
  "namespace": "biomedical",
  "endpoint": "/agents/v1/query",
  "method": "POST",
  "ip_address": "10.0.1.42",
  "user_agent": "AgentSDK/1.0",
  "request": {
    "query_hash": "sha256:def...",
    "budget": {"max_chunks": 24, "max_tokens_gen": 0},
    "allow_gen": false
  },
  "response": {
    "status_code": 200,
    "citations_count": 18,
    "degraded": false
  },
  "quota": {
    "requests_remaining": 42,
    "tokens_remaining": 0
  },
  "timings_ms": {
    "gateway": 3.2,
    "downstream": 156.4,
    "total": 159.6
  },
  "security_events": []
}
```

### Security Events

```python
class SecurityEventLogger:
    def log_quota_exceeded(self, api_key_hash: str, quota_type: str):
        """Log quota exceeded event."""
        event = {
            "event_type": "quota_exceeded",
            "api_key_hash": api_key_hash,
            "quota_type": quota_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.warning("security_event", **event)
    
    def log_permission_denied(self, api_key_hash: str, reason: str):
        """Log permission denied event."""
        event = {
            "event_type": "permission_denied",
            "api_key_hash": api_key_hash,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.warning("security_event", **event)
    
    def log_invalid_namespace(self, api_key_hash: str, namespace: str):
        """Log invalid namespace access attempt."""
        event = {
            "event_type": "invalid_namespace",
            "api_key_hash": api_key_hash,
            "namespace": namespace,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.warning("security_event", **event)
```

---

## Configuration

```yaml
agent_gateway:
  # API settings
  api:
    host: "0.0.0.0"
    port: 8080
    base_path: "/agents/v1"
    cors_origins: []  # Strict CORS, no wildcards
  
  # Authentication
  auth:
    api_key_header: "X-API-Key"
    jwt_enabled: false
    jwt_secret: "${GATEWAY_JWT_SECRET}"
    jwt_algorithms: ["HS256"]
  
  # RBAC
  rbac:
    roles:
      READER:
        max_chunks_per_request: 24
        max_tokens_per_request: 0
        max_tokens_per_day: 0
        allow_generation: false
      POWER:
        max_chunks_per_request: 48
        max_tokens_per_request: 2048
        max_tokens_per_day: 100000
        allow_generation: true
      ADMIN:
        max_chunks_per_request: 100
        max_tokens_per_request: 4096
        max_tokens_per_day: 500000
        allow_generation: true
  
  # Quotas
  quotas:
    rate_limits:
      READER: {requests_per_minute: 50, max_concurrent: 5}
      POWER: {requests_per_minute: 200, max_concurrent: 20}
      ADMIN: {requests_per_minute: 500, max_concurrent: 50}
    redis_url: "redis://localhost:6379/0"
  
  # Downstream services
  services:
    rag_service:
      url: "http://rag-service:8000"
      timeout_s: 15
    knowledge_graph:
      url: "neo4j://neo4j:7687"
      timeout_s: 10
  
  # Audit logging
  audit:
    enabled: true
    log_path: "/var/log/agent-gateway/audit.jsonl"
    redact_queries: false  # Set true in production
    retention_days: 90
  
  # Security
  security:
    strict_cors: true
    rate_limit_by_ip: true
    max_query_length: 1000
```

---

## Observability

### Metrics (Prometheus)

```
# Request counts
gateway_requests_total{endpoint="/agents/v1/query",role="READER",status="200"} 15420
gateway_requests_total{endpoint="/agents/v1/query",role="READER",status="429"} 42

# Latency
gateway_request_latency_seconds_bucket{endpoint="/agents/v1/query",le="0.2"} 14850

# Quota enforcement
gateway_quota_denials_total{reason="rate_limit"} 42
gateway_quota_denials_total{reason="token_limit"} 8

# Permission denials
gateway_permission_denials_total{reason="generation_not_allowed"} 15

# Concurrent requests
gateway_concurrent_requests{api_key_hash="sha256:abc..."} 3
```

### Structured Logs

```json
{
  "timestamp": "2025-10-23T00:00:00Z",
  "level": "INFO",
  "event": "gateway_request",
  "trace_id": "agent-query-12345",
  "endpoint": "/agents/v1/query",
  "role": "READER",
  "status_code": 200,
  "latency_ms": 159.6,
  "quota_remaining": {"requests": 42, "tokens": 0}
}
```

---

## Performance Budgets

| Operation | Target | Notes |
|-----------|--------|-------|
| Gateway overhead | ≤10ms p50 | Auth + quota checks + logging |
| Total latency | ≤300ms p50 | Gateway + RAG Service (no synthesis) |
| Quota check | ≤5ms p50 | Redis lookup |
| Audit log write | ≤20ms p50 | Async write to disk |

---

## Failure Modes

| Failure | Detection | Recovery | Response |
|---------|-----------|----------|----------|
| Quota exceeded | Redis check | N/A | 429 with `quota_remaining=0` |
| Permission denied | RBAC check | N/A | 403 with `reason` |
| Downstream degraded | RAG Service 503 | Pass-through | 503 with `degraded=true` |
| Redis unavailable | Connection error | Fail-open (log warning) | 200 (quota not enforced) |
| Downstream timeout | Timeout after budget | Return partial | 200 with `degraded=true` |

---

## Security

### CORS Policy

```python
ALLOWED_ORIGINS = []  # Strict: no wildcards, explicit origins only
ALLOWED_METHODS = ["GET", "POST"]
ALLOWED_HEADERS = ["Content-Type", "X-API-Key", "X-Trace-ID"]
EXPOSE_HEADERS = ["X-Quota-Remaining"]
```

### Data Sanitization

- **Query length limit**: 1000 characters
- **Namespace validation**: Alphanumeric + hyphen only
- **Trace ID validation**: Alphanumeric + hyphen + underscore only
- **No raw vectors exposed**: Only chunk text snippets
- **Redact sensitive data in logs**: Query text hashed in production

---

## Test Plan

### Unit Tests

1. **RBAC enforcement** (`test_rbac_roles`):
   - READER cannot set `allow_gen=true`
   - POWER can request generation
   - ADMIN can exceed default budgets

2. **Quota enforcement** (`test_quota_limits`):
   - Rate limit enforced (50 req/min for READER)
   - Token limit enforced (100k tokens/day for POWER)
   - Concurrent request cap enforced

3. **Audit logging** (`test_audit_logs`):
   - Every request logged
   - Security events captured
   - Quota remaining included

### Integration Tests

4. **End-to-end query** (`test_e2e_agent_query`):
   - Submit query as READER
   - Verify stricter budgets applied
   - Check audit log written

5. **Degraded mode propagation** (`test_degraded_passthrough`):
   - RAG Service returns `degraded=true`
   - Gateway passes through to agent
   - Audit log captures degraded flag

### Security Tests

6. **Permission bypass attempt** (`test_permission_bypass`):
   - READER tries `allow_gen=true`
   - 403 response with reason
   - Security event logged

7. **Quota exhaustion** (`test_quota_exhaustion`):
   - Send 51 requests (READER limit=50)
   - 51st request gets 429
   - Quota resets after 60s

---

## Acceptance Criteria

- ✅ Enforce RBAC (3 roles: READER, POWER, ADMIN)
- ✅ Enforce quotas (rate, tokens, concurrent)
- ✅ Audit log every request
- ✅ Pass-through degraded mode from downstream
- ✅ Gateway overhead ≤10ms p50
- ✅ All 7 tests passing (unit + integration + security)
