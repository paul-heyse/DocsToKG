# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.ratelimit.instrumentation",
#   "purpose": "Rate limiter instrumentation and telemetry helpers.",
#   "sections": [
#     {
#       "id": "normalise-service-host",
#       "name": "_normalise_service_host",
#       "anchor": "function-normalise-service-host",
#       "kind": "function"
#     },
#     {
#       "id": "emit-safe",
#       "name": "_emit_safe",
#       "anchor": "function-emit-safe",
#       "kind": "function"
#     },
#     {
#       "id": "emit-acquire-event",
#       "name": "emit_acquire_event",
#       "anchor": "function-emit-acquire-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-blocked-event",
#       "name": "emit_blocked_event",
#       "anchor": "function-emit-blocked-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-cooldown-event",
#       "name": "emit_cooldown_event",
#       "anchor": "function-emit-cooldown-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-head-skip-event",
#       "name": "emit_head_skip_event",
#       "anchor": "function-emit-head-skip-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-rate-limit-event",
#       "name": "emit_rate_limit_event",
#       "anchor": "function-emit-rate-limit-event",
#       "kind": "function"
#     },
#     {
#       "id": "emit-rate-info-event",
#       "name": "emit_rate_info_event",
#       "anchor": "function-emit-rate-info-event",
#       "kind": "function"
#     },
#     {
#       "id": "log-rate-limit-stats",
#       "name": "log_rate_limit_stats",
#       "anchor": "function-log-rate-limit-stats",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Rate limiter instrumentation and telemetry helpers.

The public API mirrors `network.polite_client` and the re-exports from
`ratelimit.__init__`. A previous refactor left stale signatures behind, which
prevented the package from importing. These helpers accept the modern
``service``/``host`` calling convention while staying defensive so older
positional usage continues to work.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from DocsToKG.OntologyDownload.observability.events import emit_event

logger = logging.getLogger(__name__)


def _normalise_service_host(
    service: Optional[str],
    host: Optional[str],
    *,
    fallback_key: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return normalised ``(service, host, key)`` tuple for event payloads."""
    if fallback_key and not service:
        parts = fallback_key.split(":", 1)
        if len(parts) == 2:
            service, host = parts
        else:
            service = parts[0]
    service_norm = (service or "default").strip().lower() or "default"
    host_norm = (host or "default").strip().lower() or "default"
    key = f"{service_norm}:{host_norm}" if host_norm != "default" else service_norm
    return service_norm, host_norm, key[:80]


def _emit_safe(
    type: str,
    *,
    level: str,
    payload: Dict[str, Any],
    service: Optional[str],
) -> None:
    """Emit the event, swallowing telemetry errors."""
    try:
        emit_event(type=type, level=level, payload=payload, service=service)
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("rate limit telemetry emission failed", exc_info=True)


def emit_acquire_event(
    key: Optional[str] = None,
    allowed: Optional[bool] = None,
    blocked_ms: Optional[float] = None,
    tokens_requested: Optional[int] = None,
    tokens_available: Optional[int] = None,
    *,
    service: Optional[str] = None,
    host: Optional[str] = None,
    weight: int = 1,
    elapsed_ms: Optional[int] = None,
) -> None:
    """Emit event when the rate limiter grants tokens."""
    service_norm, host_norm, key_norm = _normalise_service_host(service, host, fallback_key=key)
    allowed_flag = True if allowed is None else bool(allowed)
    payload: Dict[str, Any] = {
        "key": key_norm,
        "service": service_norm,
        "host": host_norm,
        "weight": max(1, int(weight)),
        "allowed": allowed_flag,
        "outcome": "allowed" if allowed_flag else "blocked",
    }
    if elapsed_ms is not None:
        payload["elapsed_ms"] = int(elapsed_ms)
    if blocked_ms is not None:
        payload["blocked_ms"] = float(blocked_ms)
    if tokens_requested is not None:
        payload["tokens_requested"] = int(tokens_requested)
    if tokens_available is not None:
        payload["tokens_available"] = int(tokens_available)

    _emit_safe(
        "ratelimit.acquire",
        level="INFO" if allowed_flag else "WARN",
        payload=payload,
        service=service_norm,
    )


def emit_blocked_event(
    *,
    service: Optional[str],
    host: Optional[str],
    weight: int,
    reason: str,
    retry_after_seconds: Optional[float] = None,
) -> None:
    """Emit event when acquisition fails due to rate limiting."""
    service_norm, host_norm, key_norm = _normalise_service_host(service, host)
    payload: Dict[str, Any] = {
        "key": key_norm,
        "service": service_norm,
        "host": host_norm,
        "weight": max(1, int(weight)),
        "reason": reason,
    }
    if retry_after_seconds is not None:
        payload["retry_after_sec"] = float(retry_after_seconds)
        payload["retry_after_ms"] = int(retry_after_seconds * 1000)

    _emit_safe(
        "ratelimit.block",
        level="WARN",
        payload=payload,
        service=service_norm,
    )


def emit_cooldown_event(
    key: Optional[str] = None,
    status_code: Optional[int] = None,
    cooldown_sec: float = 0,
    *,
    service: Optional[str] = None,
    host: Optional[str] = None,
) -> None:
    """Emit event when a cooldown (Retry-After) window activates."""
    service_norm, host_norm, key_norm = _normalise_service_host(service, host, fallback_key=key)
    payload = {
        "key": key_norm,
        "service": service_norm,
        "host": host_norm,
        "status_code": status_code,
        "cooldown_sec": float(max(0.0, cooldown_sec)),
        "cooldown_ms": int(max(0.0, cooldown_sec) * 1000),
    }
    _emit_safe(
        "ratelimit.cooldown",
        level="WARN",
        payload=payload,
        service=service_norm,
    )


def emit_head_skip_event(
    key: Optional[str] = None,
    reason: str = "unknown",
    *,
    service: Optional[str] = None,
    host: Optional[str] = None,
) -> None:
    """Emit event when a request is skipped before acquisition."""
    service_norm, host_norm, key_norm = _normalise_service_host(service, host, fallback_key=key)
    payload = {
        "key": key_norm,
        "service": service_norm,
        "host": host_norm,
        "reason": reason,
    }
    _emit_safe(
        "ratelimit.skip",
        level="INFO",
        payload=payload,
        service=service_norm,
    )


def emit_rate_limit_event(
    *,
    service: Optional[str],
    host: Optional[str],
    event: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a generic rate-limit lifecycle event."""
    service_norm, host_norm, key_norm = _normalise_service_host(service, host)
    payload = {
        "key": key_norm,
        "service": service_norm,
        "host": host_norm,
        "event": event,
    }
    if details:
        payload.update(details)
    _emit_safe(
        "ratelimit.event",
        level="INFO",
        payload=payload,
        service=service_norm,
    )


def emit_rate_info_event(
    *,
    service: Optional[str],
    host: Optional[str],
    limits: Dict[str, Any],
) -> None:
    """Emit static information about the configured rate limits."""
    service_norm, host_norm, key_norm = _normalise_service_host(service, host)
    payload = {
        "key": key_norm,
        "service": service_norm,
        "host": host_norm,
        "limits": limits,
    }
    _emit_safe(
        "ratelimit.info",
        level="INFO",
        payload=payload,
        service=service_norm,
    )


def log_rate_limit_stats(stats: Dict[str, Any]) -> None:
    """Log the latest rate limiter stats (debug-level helper)."""
    try:
        logger.debug("rate limiter stats", extra={"stats": stats})
    except Exception:  # pragma: no cover - logging shouldn't raise
        pass


__all__ = [
    "emit_acquire_event",
    "emit_blocked_event",
    "emit_cooldown_event",
    "emit_head_skip_event",
    "emit_rate_limit_event",
    "emit_rate_info_event",
    "log_rate_limit_stats",
]
