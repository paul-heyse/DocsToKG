"""Rate limiter instrumentation and telemetry.

Emits ratelimit.acquire, ratelimit.cooldown, and ratelimit.block events
for observability into rate limiting behavior and pressure.
"""

from typing import Optional

from DocsToKG.OntologyDownload.observability.events import emit_event


def emit_acquire_event(
    key: str,
    allowed: bool,
    blocked_ms: float = 0,
    tokens_requested: int = 1,
    tokens_available: int = 0,
) -> None:
    """Emit event when rate limiter acquire() is called.

    Args:
        key: Rate limit key (service or host)
        allowed: Whether the request was allowed
        blocked_ms: Milliseconds blocked if delayed
        tokens_requested: Tokens requested
        tokens_available: Tokens available after acquire
    """
    try:
        emit_event(
            type="ratelimit.acquire",
            level="INFO",
            payload={
                "key": key[:40],  # Truncate for safety
                "allowed": allowed,
                "blocked_ms": blocked_ms,
                "tokens_requested": tokens_requested,
                "tokens_available": tokens_available,
                "outcome": "allowed" if allowed else "blocked",
            },
        )
    except Exception:
        # Never fail telemetry
        pass


def emit_cooldown_event(
    key: str,
    status_code: int,
    cooldown_sec: float,
) -> None:
    """Emit event when rate limiter enters cooldown (e.g., on 429).

    Args:
        key: Rate limit key
        status_code: HTTP status code that triggered cooldown
        cooldown_sec: Seconds to wait before retrying
    """
    try:
        emit_event(
            type="ratelimit.cooldown",
            level="WARN",
            payload={
                "key": key[:40],
                "status_code": status_code,
                "cooldown_sec": cooldown_sec,
                "cooldown_ms": int(cooldown_sec * 1000),
            },
        )
    except Exception:
        # Never fail telemetry
        pass


def emit_head_skip_event(
    key: str,
    reason: str,
) -> None:
    """Emit event when rate limiter skips a request.

    Args:
        key: Rate limit key
        reason: Reason for skip (e.g., 'cooldown_active', 'no_tokens')
    """
    try:
        emit_event(
            type="ratelimit.skip",
            level="INFO",
            payload={
                "key": key[:40],
                "reason": reason,
            },
        )
    except Exception:
        # Never fail telemetry
        pass


__all__ = [
    "emit_acquire_event",
    "emit_cooldown_event",
    "emit_head_skip_event",
]
