# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.observability.queries",
#   "purpose": "Stock queries for observability analytics.",
#   "sections": [
#     {
#       "id": "get-slo-query",
#       "name": "get_slo_query",
#       "anchor": "function-get-slo-query",
#       "kind": "function"
#     },
#     {
#       "id": "get-rate-limit-query",
#       "name": "get_rate_limit_query",
#       "anchor": "function-get-rate-limit-query",
#       "kind": "function"
#     },
#     {
#       "id": "get-safety-query",
#       "name": "get_safety_query",
#       "anchor": "function-get-safety-query",
#       "kind": "function"
#     },
#     {
#       "id": "get-extraction-query",
#       "name": "get_extraction_query",
#       "anchor": "function-get-extraction-query",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Stock queries for observability analytics.

Pre-built SQL queries for DuckDB that answer common operational questions:
- SLO performance metrics (p50, p95, p99 latencies)
- Cache hit ratios by service
- Rate limiter pressure (top keys)
- Safety gate rejections (error codes)
- Zip bomb detection (compression ratios)

The module exposes a registry of named queries that the CLI can reference
without importing SQL constants directly. The registry lets us attach
categories, descriptions, and other metadata while keeping SQL definitions
co-located for easier maintenance.
"""

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class QueryDefinition:
    """Container describing a stock query."""

    name: str
    sql: str
    description: str
    category: str


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
# Query Registry / API
# ============================================================================

_QUERY_DEFINITIONS: Dict[str, QueryDefinition] = {
    "net_latency_distribution": QueryDefinition(
        name="net_latency_distribution",
        sql=QUERY_SLO_NETWORK_LATENCY,
        description=(
            "Latency percentiles (p50/p95/p99) and extrema for network requests "
            "grouped by service."
        ),
        category="slo",
    ),
    "net_cache_hit_ratio": QueryDefinition(
        name="net_cache_hit_ratio",
        sql=QUERY_CACHE_HIT_RATIO,
        description="Cache hit ratios for HTTP requests grouped by service.",
        category="slo",
    ),
    "ratelimit_pressure": QueryDefinition(
        name="ratelimit_pressure",
        sql=QUERY_RATE_LIMIT_PRESSURE,
        description=(
            "Top rate-limiter keys experiencing blocking along with block rates "
            "and cumulative blocked milliseconds."
        ),
        category="ratelimit",
    ),
    "ratelimit_cooldowns": QueryDefinition(
        name="ratelimit_cooldowns",
        sql=QUERY_RATE_LIMIT_COOLDOWNS,
        description="Cooldown windows triggered by 429 responses per key.",
        category="ratelimit",
    ),
    "policy_rejections": QueryDefinition(
        name="policy_rejections",
        sql=QUERY_POLICY_GATE_REJECTIONS,
        description="Policy gate error codes ranked by rejection count.",
        category="policy",
    ),
    "policy_heatmap": QueryDefinition(
        name="policy_heatmap",
        sql=QUERY_SAFETY_HEATMAP,
        description="Daily error trends by event type for policy/safety issues.",
        category="policy",
    ),
    "extract_zip_bombs": QueryDefinition(
        name="extract_zip_bombs",
        sql=QUERY_ZIP_BOMB_SENTINEL,
        description=(
            "Detect potential zip bombs using entry counts, bytes per entry, and "
            "extraction durations."
        ),
        category="extraction",
    ),
    "extract_summary": QueryDefinition(
        name="extract_summary",
        sql=QUERY_EXTRACTION_STATS,
        description="Daily aggregate statistics for extraction jobs.",
        category="extraction",
    ),
}


def list_queries(category: str | None = None) -> list[str]:
    """Return the list of registered query names.

    Args:
        category: Optional category filter (e.g. ``"slo"`` or ``"ratelimit"``).

    Returns:
        Sorted list of query names matching the filter.
    """

    names: Iterable[str] = (
        definition.name
        for definition in _QUERY_DEFINITIONS.values()
        if category is None or definition.category == category
    )
    return sorted(set(names))


def get_query(name: str) -> str:
    """Retrieve the SQL for a registered query.

    Args:
        name: Name of the query to load.

    Returns:
        SQL string belonging to the named query.

    Raises:
        KeyError: If ``name`` is not registered.
    """

    try:
        return _QUERY_DEFINITIONS[name].sql
    except KeyError as exc:  # pragma: no cover - defensive error path
        raise KeyError(name) from exc


def query_summary(category: str | None = None) -> Dict[str, str]:
    """Return descriptions for registered queries.

    Args:
        category: Optional category filter.

    Returns:
        Mapping of query name to human-readable description.
    """

    return {
        definition.name: (
            f"{definition.description} (category: {definition.category})"
        )
        for definition in _QUERY_DEFINITIONS.values()
        if category is None or definition.category == category
    }


# ---------------------------------------------------------------------------
# Legacy helpers retained for backwards compatibility
# ---------------------------------------------------------------------------


def get_slo_query(metric: str = "network") -> str | None:
    """Get SLO query by metric name.

    Args:
        metric: 'network' for latency, 'cache' for hit ratio

    Returns:
        SQL query string or None
    """

    mapping = {
        "network": "net_latency_distribution",
        "cache": "net_cache_hit_ratio",
    }
    name = mapping.get(metric)
    return get_query(name) if name else None


def get_rate_limit_query(metric: str = "pressure") -> str | None:
    """Get rate limiting query by metric name.

    Args:
        metric: 'pressure' for blocked keys, 'cooldowns' for 429s

    Returns:
        SQL query string or None
    """

    mapping = {
        "pressure": "ratelimit_pressure",
        "cooldowns": "ratelimit_cooldowns",
    }
    name = mapping.get(metric)
    return get_query(name) if name else None


def get_safety_query(metric: str = "rejections") -> str | None:
    """Get safety/policy query by metric name.

    Args:
        metric: 'rejections' for error codes, 'heatmap' for timeline

    Returns:
        SQL query string or None
    """

    mapping = {
        "rejections": "policy_rejections",
        "heatmap": "policy_heatmap",
    }
    name = mapping.get(metric)
    return get_query(name) if name else None


def get_extraction_query(metric: str = "bombs") -> str | None:
    """Get extraction query by metric name.

    Args:
        metric: 'bombs' for zip bomb detection, 'stats' for summary

    Returns:
        SQL query string or None
    """

    mapping = {
        "bombs": "extract_zip_bombs",
        "stats": "extract_summary",
    }
    name = mapping.get(metric)
    return get_query(name) if name else None


__all__ = [
    "QueryDefinition",
    "QUERY_SLO_NETWORK_LATENCY",
    "QUERY_CACHE_HIT_RATIO",
    "QUERY_RATE_LIMIT_PRESSURE",
    "QUERY_RATE_LIMIT_COOLDOWNS",
    "QUERY_POLICY_GATE_REJECTIONS",
    "QUERY_SAFETY_HEATMAP",
    "QUERY_ZIP_BOMB_SENTINEL",
    "QUERY_EXTRACTION_STATS",
    "list_queries",
    "get_query",
    "query_summary",
    "get_slo_query",
    "get_rate_limit_query",
    "get_safety_query",
    "get_extraction_query",
]
