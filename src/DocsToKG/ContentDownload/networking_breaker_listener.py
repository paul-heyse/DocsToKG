# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.networking_breaker_listener",
#   "purpose": "Telemetry listener for circuit breaker state transitions",
#   "sections": [
#     {
#       "id": "breakertelemetrysink",
#       "name": "BreakerTelemetrySink",
#       "anchor": "class-breakertelemetrysink",
#       "kind": "class"
#     },
#     {
#       "id": "breakerlistenerconfig",
#       "name": "BreakerListenerConfig",
#       "anchor": "class-breakerlistenerconfig",
#       "kind": "class"
#     },
#     {
#       "id": "networkbreakerlistener",
#       "name": "NetworkBreakerListener",
#       "anchor": "class-networkbreakerlistener",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Telemetry listener for circuit breaker state transitions.

This module provides a pybreaker listener that emits structured telemetry
events for circuit breaker state changes, failures, and successes.

Typical Usage:
    from DocsToKG.ContentDownload.networking_breaker_listener import (
        NetworkBreakerListener, BreakerListenerConfig
    )

    listener = NetworkBreakerListener(telemetry_sink, BreakerListenerConfig(
        run_id="run-123", host="api.crossref.org", scope="host"
    ))

    # Attach to pybreaker instance
    breaker = pybreaker.CircuitBreaker(listeners=[listener])
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

try:
    import pybreaker
except ImportError:  # pragma: no cover
    pybreaker = None  # type: ignore


class BreakerTelemetrySink(Protocol):
    """Protocol for emitting breaker telemetry events."""

    def emit(self, event: Mapping[str, Any]) -> None: ...


@dataclass
class BreakerListenerConfig:
    """Configuration for a breaker listener."""

    run_id: str
    host: str  # breaker key (host)
    scope: str = "host"  # "host" | "resolver"
    resolver: str | None = None


class NetworkBreakerListener(pybreaker.CircuitBreakerListener):  # type: ignore[misc]
    """Emits state transitions & per-call signals for a single breaker."""

    def __init__(self, sink: BreakerTelemetrySink, cfg: BreakerListenerConfig):
        self.sink = sink
        self.cfg = cfg

    def _emit(self, event_type: str, **body: Any) -> None:
        payload = {
            "event_type": f"breaker_{event_type}",
            "ts": time.time(),
            "run_id": self.cfg.run_id,
            "host": self.cfg.host,
            "scope": self.cfg.scope,
            "resolver": self.cfg.resolver,
        }
        payload.update(body)
        self.sink.emit(payload)

    # Called right before the protected call executes
    def before_call(self, cb, func, *args, **kwargs):
        self._emit("before_call", state=str(cb.current_state))

    # Called when the protected call succeeds
    def success(self, cb):
        self._emit(
            "success", state=str(cb.current_state), fail_counter=getattr(cb, "fail_counter", None)
        )

    # Called when the protected call fails (exception raised by the call)
    def failure(self, cb, exc):
        self._emit(
            "failure",
            state=str(cb.current_state),
            exc_type=type(exc).__name__,
            msg=str(exc)[:300],
            fail_counter=getattr(cb, "fail_counter", None),
        )

    # Called when circuit breaker state changes
    def state_change(self, cb, old_state, new_state):
        # States are classes; stringify for logs
        self._emit(
            "state_change",
            old=str(old_state),
            new=str(new_state),
            reset_timeout_s=getattr(cb, "reset_timeout", None),
        )
