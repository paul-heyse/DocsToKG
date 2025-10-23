# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.ratelimit.__init__",
#   "purpose": "Rate-limiting subsystem: Multi-window rate-limiting with pyrate-limiter.",
#   "sections": []
# }
# === /NAVMAP ===

"""Rate-limiting subsystem: Multi-window rate-limiting with pyrate-limiter.

This package provides a production-ready rate-limiting façade based on pyrate-limiter,
a battle-tested library for enforcing request quotas across multiple time windows.

Architecture:
- Single shared RateLimitManager instance (singleton)
- Per-service rate limits (configurable per API)
- Multi-window enforcement (e.g., 5/sec AND 300/min simultaneously)
- Cross-process coordination (SQLiteBucket for shared state)
- Weighted requests (some requests consume multiple slots)
- Block vs fail-fast modes

Modules:
- config: RateSpec parsing, validation, normalization
- manager: RateLimitManager façade with acquire() semantics
- instrumentation: Rate-limit event telemetry

Example:
    >>> from DocsToKG.OntologyDownload.ratelimit import get_rate_limiter
    >>> limiter = get_rate_limiter()
    >>> limiter.acquire("ols", "www.ebi.ac.uk", weight=1)
    >>> # Blocks until slot available (respects all rate windows)
"""

from DocsToKG.OntologyDownload.ratelimit.config import (
    RateSpec,
    get_schema_summary,
    normalize_per_service_rates,
    normalize_rate_list,
    parse_rate_string,
    validate_rate_list,
)
from DocsToKG.OntologyDownload.ratelimit.instrumentation import (
    emit_acquire_event,
    emit_blocked_event,
    emit_rate_info_event,
    emit_rate_limit_event,
    log_rate_limit_stats,
)
from DocsToKG.OntologyDownload.ratelimit.manager import (
    RateLimitManager,
    close_rate_limiter,
    get_rate_limiter,
    reset_rate_limiter,
)

__all__ = [
    # Config
    "RateSpec",
    "parse_rate_string",
    "validate_rate_list",
    "normalize_rate_list",
    "normalize_per_service_rates",
    "get_schema_summary",
    # Manager
    "RateLimitManager",
    "get_rate_limiter",
    "close_rate_limiter",
    "reset_rate_limiter",
    # Instrumentation
    "emit_rate_limit_event",
    "emit_acquire_event",
    "emit_blocked_event",
    "emit_rate_info_event",
    "log_rate_limit_stats",
]
