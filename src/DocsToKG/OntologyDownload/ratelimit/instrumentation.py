"""Rate-limiting instrumentation: Structured telemetry for rate-limit events.

Emits structured events when rate limits are acquired, blocked, or released.
Events are emitted to Python logging with contextual information for
observability and debugging.

Event types:
- ratelimit.acquire: Successfully acquired rate limit slot(s)
- ratelimit.blocked: Rate limit exceeded, request blocked/delayed
- ratelimit.release: Slots released (when available again)

Example:
    >>> from DocsToKG.OntologyDownload.ratelimit.instrumentation import emit_rate_limit_event
    >>> emit_rate_limit_event("acquire", {
    ...     "service": "ols",
    ...     "host": "www.ebi.ac.uk",
    ...     "weight": 1,
    ... })
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Event Emission
# ============================================================================


def emit_rate_limit_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Emit a structured rate-limit event.

    Events are emitted to Python logging with structured context.
    Callers should configure logging to route events to observability stack.

    Args:
        event_type: Event type (acquire, blocked, release, etc.)
        payload: Dictionary of event fields (no secrets)

    Example:
        >>> emit_rate_limit_event("ratelimit.acquire", {
        ...     "service": "ols",
        ...     "host": "example.com",
        ...     "weight": 1,
        ...     "elapsed_ms": 5,
        ... })
    """
    full_event_type = f"ratelimit.{event_type}"
    logger.info(
        full_event_type,
        extra={
            "event_type": full_event_type,
            **payload,  # Spread all fields as log context
        },
    )


# ============================================================================
# High-Level Event Helpers
# ============================================================================


def emit_acquire_event(
    service: str,
    host: Optional[str] = None,
    weight: int = 1,
    elapsed_ms: int = 0,
) -> None:
    """Emit a rate-limit acquire event.

    Called when rate limit slots are successfully acquired.

    Args:
        service: Service name
        host: Optional host name
        weight: Number of slots acquired
        elapsed_ms: Milliseconds spent waiting (0 if immediate)
    """
    payload = {
        "service": service,
        "weight": weight,
        "elapsed_ms": elapsed_ms,
    }
    if host:
        payload["host"] = host

    emit_rate_limit_event("acquire", payload)


def emit_blocked_event(
    service: str,
    host: Optional[str] = None,
    weight: int = 1,
    reason: str = "limit_exceeded",
) -> None:
    """Emit a rate-limit blocked event.

    Called when a request is blocked due to rate limit.

    Args:
        service: Service name
        host: Optional host name
        weight: Number of slots that would have been needed
        reason: Reason for blocking (limit_exceeded, max_delay_exceeded, etc.)
    """
    payload = {
        "service": service,
        "weight": weight,
        "reason": reason,
    }
    if host:
        payload["host"] = host

    emit_rate_limit_event("blocked", payload)


def emit_rate_info_event(
    service: str,
    rates: List[str],
    mode: str = "block",
) -> None:
    """Emit rate configuration info event.

    Called when a rate limiter is created or updated.

    Args:
        service: Service name
        rates: List of rate spec strings (e.g., ["4/second", "300/minute"])
        mode: Rate limiting mode (block, fail-fast, etc.)
    """
    payload = {
        "service": service,
        "rates": rates,
        "mode": mode,
    }
    emit_rate_limit_event("info", payload)


# ============================================================================
# Context-Aware Helpers
# ============================================================================


def log_rate_limit_stats(service: str, stats: Dict[str, Any]) -> None:
    """Log rate-limiting statistics for a service.

    Args:
        service: Service name
        stats: Statistics dictionary (from manager.get_stats())
    """
    logger.debug(
        "Rate limit statistics",
        extra={
            "event_type": "ratelimit.stats",
            "service": service,
            **stats,
        },
    )


__all__ = [
    "emit_rate_limit_event",
    "emit_acquire_event",
    "emit_blocked_event",
    "emit_rate_info_event",
    "log_rate_limit_stats",
]
