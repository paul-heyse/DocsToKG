"""Stock SQL queries for operational dashboards and analytics.

Provides pre-built queries for answering common operational questions:
- SLO metrics (latencies, error rates)
- Cache performance (hit ratios)
- Rate-limit pressure (blocked time)
- Safety gates (policy rejections)
- Capacity trends (bytes, entries)
"""

from typing import Dict, List

# ============================================================================
# Query Registry (for documentation)
# ============================================================================

STOCK_QUERIES: Dict[str, Dict[str, str]] = {}


def register_query(name: str, description: str):
    """Decorator to register a query for documentation."""

    def decorator(query: str) -> str:
        STOCK_QUERIES[name] = {"query": query, "description": description}
        return query

    return decorator


# ============================================================================
# SLO Metrics Queries
# ============================================================================


QUERY_NET_REQUEST_P95_LATENCY = """
SELECT
    service,
    APPROX_QUANTILE(payload.elapsed_ms, 0.95) as p95_latency_ms,
    APPROX_QUANTILE(payload.elapsed_ms, 0.50) as p50_latency_ms,
    MAX(payload.elapsed_ms) as max_latency_ms,
    COUNT(*) as request_count
FROM events
WHERE type = 'net.request'
    AND ts >= now() - interval '1 hour'
GROUP BY service
ORDER BY p95_latency_ms DESC;
"""
STOCK_QUERIES["net_request_p95_latency"] = {
    "description": "Network request SLO: p95/p50 latencies by service (last hour)",
    "query": QUERY_NET_REQUEST_P95_LATENCY,
}


QUERY_ERROR_RATE_BY_SERVICE = """
SELECT
    service,
    COUNT(*) FILTER (WHERE level = 'ERROR') as error_count,
    COUNT(*) as total_events,
    ROUND(100.0 * COUNT(*) FILTER (WHERE level = 'ERROR') / COUNT(*), 2) as error_rate_pct
FROM events
WHERE ts >= now() - interval '1 hour'
GROUP BY service
ORDER BY error_rate_pct DESC;
"""
STOCK_QUERIES["error_rate_by_service"] = {
    "description": "Error rate by service (last hour)",
    "query": QUERY_ERROR_RATE_BY_SERVICE,
}


# ============================================================================
# Cache Performance Queries
# ============================================================================


QUERY_CACHE_HIT_RATIO = """
SELECT
    service,
    COUNT(*) FILTER (WHERE payload.cache IN ('hit', 'revalidated')) as cache_hits,
    COUNT(*) as total_requests,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE payload.cache IN ('hit', 'revalidated')) / COUNT(*),
        2
    ) as cache_hit_ratio_pct
FROM events
WHERE type = 'net.request'
    AND ts >= now() - interval '1 hour'
GROUP BY service
ORDER BY cache_hit_ratio_pct DESC;
"""
STOCK_QUERIES["cache_hit_ratio"] = {
    "description": "HTTP cache hit ratio by service (last hour)",
    "query": QUERY_CACHE_HIT_RATIO,
}


# ============================================================================
# Rate-Limit Pressure Queries
# ============================================================================


QUERY_RATE_LIMIT_PRESSURE = """
SELECT
    SUBSTR(payload.key, 1, 40) as key,
    SUM(payload.blocked_ms) as total_blocked_ms,
    COUNT(*) as acquire_attempts,
    ROUND(100.0 * SUM(payload.blocked_ms) / NULLIF(SUM(payload.elapsed_ms), 0), 2) as blocked_pct
FROM events
WHERE type = 'ratelimit.acquire'
    AND ts >= now() - interval '1 hour'
GROUP BY key
ORDER BY total_blocked_ms DESC
LIMIT 20;
"""
STOCK_QUERIES["rate_limit_pressure"] = {
    "description": "Top 20 rate-limited keys by blocked time (last hour)",
    "query": QUERY_RATE_LIMIT_PRESSURE,
}


QUERY_429_RESPONSES = """
SELECT
    payload.host,
    COUNT(*) as count_429,
    AVG(payload.retry_after_s) as avg_retry_after_s,
    MAX(payload.retry_after_s) as max_retry_after_s
FROM events
WHERE type = 'net.request'
    AND payload.status = 429
    AND ts >= now() - interval '1 hour'
GROUP BY payload.host
ORDER BY count_429 DESC;
"""
STOCK_QUERIES["http_429_responses"] = {
    "description": "HTTP 429 responses by host (last hour)",
    "query": QUERY_429_RESPONSES,
}


# ============================================================================
# Safety Gate Queries
# ============================================================================


QUERY_POLICY_REJECTIONS = """
SELECT
    payload.error_code,
    COUNT(*) as rejection_count,
    COUNT(DISTINCT run_id) as distinct_runs
FROM events
WHERE type LIKE '%.error'
    AND payload.error_code LIKE 'E_%'
    AND ts >= now() - interval '1 hour'
GROUP BY payload.error_code
ORDER BY rejection_count DESC;
"""
STOCK_QUERIES["policy_rejections"] = {
    "description": "Policy gate rejections by error code (last hour)",
    "query": QUERY_POLICY_REJECTIONS,
}


QUERY_EXTRACTION_FAILURES = """
SELECT
    payload.error_code,
    COUNT(*) as failure_count,
    AVG(payload.ratio_total) as avg_ratio,
    MAX(payload.ratio_total) as max_ratio
FROM events
WHERE type = 'extract.error'
    AND ts >= now() - interval '24 hours'
GROUP BY payload.error_code
ORDER BY failure_count DESC;
"""
STOCK_QUERIES["extraction_failures"] = {
    "description": "Extraction failures by error code (last 24 hours)",
    "query": QUERY_EXTRACTION_FAILURES,
}


# ============================================================================
# Capacity & Throughput Queries
# ============================================================================


QUERY_BYTES_WRITTEN_BY_SERVICE = """
SELECT
    service,
    SUM(payload.bytes_written) as total_bytes_written,
    COUNT(*) as extract_jobs,
    ROUND(SUM(payload.bytes_written) / 1024.0 / 1024.0, 2) as total_mb
FROM events
WHERE type = 'extract.done'
    AND ts >= now() - interval '1 hour'
GROUP BY service
ORDER BY total_bytes_written DESC;
"""
STOCK_QUERIES["bytes_written_by_service"] = {
    "description": "Throughput: bytes written by service (last hour)",
    "query": QUERY_BYTES_WRITTEN_BY_SERVICE,
}


QUERY_EXTRACTION_STATS = """
SELECT
    COUNT(*) as total_extracts,
    AVG(payload.entries_total) as avg_entries,
    AVG(payload.duration_ms) as avg_duration_ms,
    AVG(payload.ratio_total) as avg_compression_ratio,
    MAX(payload.ratio_total) as max_ratio_hint
FROM events
WHERE type = 'extract.done'
    AND ts >= now() - interval '1 hour';
"""
STOCK_QUERIES["extraction_stats"] = {
    "description": "Aggregate extraction statistics (last hour)",
    "query": QUERY_EXTRACTION_STATS,
}


# ============================================================================
# Anomaly Detection Queries
# ============================================================================


QUERY_ZIP_BOMB_CANDIDATES = """
SELECT
    ts,
    payload.ratio_total,
    payload.entries_total,
    payload.bytes_declared,
    payload.bytes_written
FROM events
WHERE type = 'extract.done'
    AND payload.ratio_total > 10.0
    AND ts >= now() - interval '7 days'
ORDER BY payload.ratio_total DESC
LIMIT 20;
"""
STOCK_QUERIES["zip_bomb_candidates"] = {
    "description": "Potential zip-bomb detections (high compression ratio > 10.0)",
    "query": QUERY_ZIP_BOMB_CANDIDATES,
}


QUERY_SLOW_REQUESTS = """
SELECT
    ts,
    payload.url_redacted,
    payload.host,
    payload.elapsed_ms,
    payload.status
FROM events
WHERE type = 'net.request'
    AND payload.elapsed_ms > 30000
    AND ts >= now() - interval '24 hours'
ORDER BY payload.elapsed_ms DESC
LIMIT 20;
"""
STOCK_QUERIES["slow_requests"] = {
    "description": "Slow HTTP requests (> 30s) in last 24 hours",
    "query": QUERY_SLOW_REQUESTS,
}


# ============================================================================
# Query Functions
# ============================================================================


def get_stock_queries() -> Dict[str, Dict[str, str]]:
    """Get all registered stock queries.

    Returns:
        Dict mapping query name to {description, query}
    """
    return STOCK_QUERIES.copy()


def get_query(name: str) -> str:
    """Get a stock query by name.

    Args:
        name: Query name (e.g., 'net_request_p95_latency')

    Returns:
        SQL query string or raises KeyError if not found

    Raises:
        KeyError: If query name not found
    """
    if name not in STOCK_QUERIES:
        available = ", ".join(sorted(STOCK_QUERIES.keys()))
        raise KeyError(f"Query '{name}' not found. Available: {available}")
    return STOCK_QUERIES[name]["query"]


def list_queries() -> List[str]:
    """List all available stock query names.

    Returns:
        Sorted list of query names
    """
    return sorted(STOCK_QUERIES.keys())


def describe_query(name: str) -> str:
    """Get human-readable description of a query.

    Args:
        name: Query name

    Returns:
        Description string or raises KeyError if not found
    """
    if name not in STOCK_QUERIES:
        raise KeyError(f"Query '{name}' not found")
    return STOCK_QUERIES[name]["description"]


def query_summary() -> Dict[str, str]:
    """Get a summary of all queries.

    Returns:
        Dict mapping query name to description
    """
    return {name: info["description"] for name, info in STOCK_QUERIES.items()}


__all__ = [
    "STOCK_QUERIES",
    "get_stock_queries",
    "get_query",
    "list_queries",
    "describe_query",
    "query_summary",
    # Query constants for direct access
    "QUERY_NET_REQUEST_P95_LATENCY",
    "QUERY_ERROR_RATE_BY_SERVICE",
    "QUERY_CACHE_HIT_RATIO",
    "QUERY_RATE_LIMIT_PRESSURE",
    "QUERY_429_RESPONSES",
    "QUERY_POLICY_REJECTIONS",
    "QUERY_EXTRACTION_FAILURES",
    "QUERY_BYTES_WRITTEN_BY_SERVICE",
    "QUERY_EXTRACTION_STATS",
    "QUERY_ZIP_BOMB_CANDIDATES",
    "QUERY_SLOW_REQUESTS",
]
