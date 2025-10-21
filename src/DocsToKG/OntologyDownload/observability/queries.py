"""Stock queries for observability analytics.

Pre-built SQL queries for DuckDB that answer common operational questions:
- SLO performance metrics (p50, p95, p99 latencies)
- Cache hit ratios by service
- Rate limiter pressure (top keys)
- Safety gate rejections (error codes)
- Zip bomb detection (compression ratios)
"""

from typing import Optional

# ============================================================================
# SLO Queries (Performance)
# ============================================================================

QUERY_SLO_NETWORK_LATENCY = """
SELECT
    service,
    COUNT(*) as request_count,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY payload.elapsed_ms) as p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY payload.elapsed_ms) as p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY payload.elapsed_ms) as p99_ms,
    AVG(payload.elapsed_ms) as avg_ms,
    MAX(payload.elapsed_ms) as max_ms
FROM events
WHERE type = 'net.request'
GROUP BY service
ORDER BY p95_ms DESC
"""

QUERY_CACHE_HIT_RATIO = """
SELECT
    service,
    COUNT(*) as total_requests,
    SUM(CASE WHEN payload.cache IN ('hit', 'revalidated') THEN 1 ELSE 0 END) as cached_requests,
    ROUND(100.0 * SUM(CASE WHEN payload.cache IN ('hit', 'revalidated') THEN 1 ELSE 0 END) / COUNT(*), 2) as hit_ratio_percent
FROM events
WHERE type = 'net.request'
GROUP BY service
ORDER BY hit_ratio_percent DESC
"""


# ============================================================================
# Rate Limiting Queries
# ============================================================================

QUERY_RATE_LIMIT_PRESSURE = """
SELECT
    SUBSTR(payload.key, 1, 40) as key,
    COUNT(*) as acquire_count,
    SUM(CASE WHEN payload.allowed = false THEN 1 ELSE 0 END) as blocked_count,
    SUM(payload.blocked_ms) as total_blocked_ms,
    ROUND(100.0 * SUM(CASE WHEN payload.allowed = false THEN 1 ELSE 0 END) / COUNT(*), 2) as block_rate_percent
FROM events
WHERE type = 'ratelimit.acquire'
GROUP BY SUBSTR(payload.key, 1, 40)
HAVING SUM(payload.blocked_ms) > 0
ORDER BY total_blocked_ms DESC
LIMIT 10
"""

QUERY_RATE_LIMIT_COOLDOWNS = """
SELECT
    SUBSTR(payload.key, 1, 40) as key,
    payload.status_code,
    COUNT(*) as cooldown_count,
    SUM(payload.cooldown_ms) as total_cooldown_ms,
    AVG(payload.cooldown_sec) as avg_cooldown_sec
FROM events
WHERE type = 'ratelimit.cooldown'
GROUP BY SUBSTR(payload.key, 1, 40), payload.status_code
ORDER BY cooldown_count DESC
"""


# ============================================================================
# Safety & Policy Queries
# ============================================================================

QUERY_POLICY_GATE_REJECTIONS = """
SELECT
    payload.error_code,
    COUNT(*) as rejection_count,
    ROUND(AVG(payload.elapsed_ms), 2) as avg_gate_ms,
    MAX(payload.elapsed_ms) as max_gate_ms
FROM events
WHERE type LIKE '%.error'
GROUP BY payload.error_code
ORDER BY rejection_count DESC
"""

QUERY_SAFETY_HEATMAP = """
SELECT
    ts::DATE as date,
    type,
    COUNT(*) as error_count
FROM events
WHERE type LIKE '%.error'
GROUP BY ts::DATE, type
ORDER BY ts DESC, error_count DESC
LIMIT 50
"""


# ============================================================================
# Extraction Queries
# ============================================================================

QUERY_ZIP_BOMB_SENTINEL = """
SELECT
    ts,
    service,
    payload.entries_total,
    payload.entries_included,
    ROUND(payload.bytes_written / NULLIF(payload.entries_included, 0), 2) as bytes_per_entry,
    payload.duration_ms
FROM events
WHERE type = 'extract.done'
    AND (payload.entries_total > 1000 OR ROUND(payload.bytes_written / NULLIF(payload.entries_included, 0), 2) > 10.0)
ORDER BY ts DESC
LIMIT 20
"""

QUERY_EXTRACTION_STATS = """
SELECT
    DATE(ts) as date,
    COUNT(*) as extraction_count,
    SUM(payload.entries_total) as total_entries,
    SUM(payload.bytes_written) as total_bytes,
    AVG(payload.duration_ms) as avg_duration_ms
FROM events
WHERE type = 'extract.done'
GROUP BY DATE(ts)
ORDER BY date DESC
"""


# ============================================================================
# Query Functions
# ============================================================================


def get_slo_query(metric: str = "network") -> Optional[str]:
    """Get SLO query by metric name.

    Args:
        metric: 'network' for latency, 'cache' for hit ratio

    Returns:
        SQL query string or None
    """
    queries = {
        "network": QUERY_SLO_NETWORK_LATENCY,
        "cache": QUERY_CACHE_HIT_RATIO,
    }
    return queries.get(metric)


def get_rate_limit_query(metric: str = "pressure") -> Optional[str]:
    """Get rate limiting query by metric name.

    Args:
        metric: 'pressure' for blocked keys, 'cooldowns' for 429s

    Returns:
        SQL query string or None
    """
    queries = {
        "pressure": QUERY_RATE_LIMIT_PRESSURE,
        "cooldowns": QUERY_RATE_LIMIT_COOLDOWNS,
    }
    return queries.get(metric)


def get_safety_query(metric: str = "rejections") -> Optional[str]:
    """Get safety/policy query by metric name.

    Args:
        metric: 'rejections' for error codes, 'heatmap' for timeline

    Returns:
        SQL query string or None
    """
    queries = {
        "rejections": QUERY_POLICY_GATE_REJECTIONS,
        "heatmap": QUERY_SAFETY_HEATMAP,
    }
    return queries.get(metric)


def get_extraction_query(metric: str = "bombs") -> Optional[str]:
    """Get extraction query by metric name.

    Args:
        metric: 'bombs' for zip bomb detection, 'stats' for summary

    Returns:
        SQL query string or None
    """
    queries = {
        "bombs": QUERY_ZIP_BOMB_SENTINEL,
        "stats": QUERY_EXTRACTION_STATS,
    }
    return queries.get(metric)


__all__ = [
    "QUERY_SLO_NETWORK_LATENCY",
    "QUERY_CACHE_HIT_RATIO",
    "QUERY_RATE_LIMIT_PRESSURE",
    "QUERY_RATE_LIMIT_COOLDOWNS",
    "QUERY_POLICY_GATE_REJECTIONS",
    "QUERY_SAFETY_HEATMAP",
    "QUERY_ZIP_BOMB_SENTINEL",
    "QUERY_EXTRACTION_STATS",
    "get_slo_query",
    "get_rate_limit_query",
    "get_safety_query",
    "get_extraction_query",
]
