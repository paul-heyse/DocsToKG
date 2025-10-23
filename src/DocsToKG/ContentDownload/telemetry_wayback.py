# {
#   "module": "DocsToKG.ContentDownload.telemetry_wayback",
#   "purpose": "Wayback resolver telemetry helpers.",
#   "sections": [
#     {
#       "id": "ensure-event-buffer",
#       "name": "_ensure_event_buffer",
#       "anchor": "function-ensure-event-buffer",
#       "kind": "function"
#     },
#     {
#       "id": "telemetrywaybackattempt",
#       "name": "TelemetryWaybackAttempt",
#       "anchor": "class-telemetrywaybackattempt",
#       "kind": "class"
#     },
#     {
#       "id": "telemetrywayback",
#       "name": "TelemetryWayback",
#       "anchor": "class-telemetrywayback",
#       "kind": "class"
#     },
#     {
#       "id": "telemetryattemptctx",
#       "name": "_TelemetryAttemptCtx",
#       "anchor": "class-telemetryattemptctx",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Wayback resolver telemetry helpers."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


def _ensure_event_buffer(ctx: Any) -> list[dict[str, Any]] | None:
    """Return an event buffer stored on ``ctx`` if telemetry is enabled."""

    if ctx is None:
        return None

    buffer = getattr(ctx, "telemetry_wayback_events", None)
    if buffer is None:
        buffer = []
        try:
            ctx.telemetry_wayback_events = buffer
        except Exception:  # pragma: no cover - defensive guard
            return None
    return buffer


@dataclass(slots=True)
class TelemetryWaybackAttempt:
    """State container for a single Wayback attempt."""

    _telemetry: "TelemetryWayback" = field(repr=False)
    _ctx: Any = field(repr=False)
    attempt_id: str
    original_url: str
    canonical_url: str
    status: str = "pending"
    candidate_url: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    _completed: bool = field(default=False, init=False, repr=False)

    def record_discovery(self, metadata: Mapping[str, Any]) -> None:
        """Record discovery metadata for this attempt."""

        self._append_event("discovery", metadata=dict(metadata))

    def record_candidate(
        self,
        outcome: str,
        *,
        candidate_url: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        """Record a candidate emission event."""

        self.candidate_url = candidate_url
        self._append_event(
            "candidate",
            outcome=outcome,
            candidate_url=candidate_url,
            metadata=dict(metadata),
        )

    def record_error(
        self,
        *,
        error: str,
        error_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an error event for the attempt."""

        payload = {"error": error}
        if error_type:
            payload["error_type"] = error_type
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        self._append_event("error", **payload)

    def complete(self, status: str, metadata: Mapping[str, Any] | None = None) -> None:
        """Mark the attempt as completed with ``status``."""

        if self._completed:
            return

        metadata_copy = dict(metadata) if metadata else {}
        if metadata_copy:
            self.extra.update(metadata_copy)

        payload = {
            "candidate_url": self.candidate_url,
        }
        if metadata_copy:
            payload["metadata"] = metadata_copy

        self.status = status
        self._append_event("complete", status=status, **payload)
        self._completed = True

    def as_metadata(self) -> dict[str, Any]:
        """Return structured metadata for embedding into resolver results."""

        return {
            "attempt_id": self.attempt_id,
            "original_url": self.original_url,
            "canonical_url": self.canonical_url,
            "status": self.status,
            "candidate_url": self.candidate_url,
            "extra": deepcopy(self.extra),
            "events": deepcopy(self.events),
        }

    def ensure_completed(self) -> None:
        """Ensure attempts that exit early are marked as aborted."""

        if not self._completed:
            self.complete("aborted")

    def _append_event(self, event: str, **payload: Any) -> None:
        event_payload = {"event": event, **payload}
        self.events.append(deepcopy(event_payload))

        buffer = _ensure_event_buffer(self._ctx)
        if buffer is not None:
            buffer.append(
                {
                    "attempt_id": self.attempt_id,
                    "original_url": self.original_url,
                    "canonical_url": self.canonical_url,
                    **deepcopy(event_payload),
                }
            )


class TelemetryWayback:
    """Factory for Wayback attempt telemetry contexts."""

    def start_attempt(
        self,
        ctx: Any,
        *,
        original_url: str,
        canonical_url: str,
    ) -> "_TelemetryAttemptCtx":
        """Return context manager for telemetry-wrapped attempts."""

        return _TelemetryAttemptCtx(self, ctx, original_url, canonical_url)


class _TelemetryAttemptCtx:
    """Context manager returned by :class:`TelemetryWayback`."""

    def __init__(
        self,
        telemetry: TelemetryWayback,
        ctx: Any,
        original_url: str,
        canonical_url: str,
    ) -> None:
        self._telemetry = telemetry
        self._ctx = ctx
        self._original_url = original_url
        self._canonical_url = canonical_url
        self._attempt: TelemetryWaybackAttempt | None = None

    def __enter__(self) -> TelemetryWaybackAttempt:
        attempt = TelemetryWaybackAttempt(
            self._telemetry,
            self._ctx,
            attempt_id=uuid.uuid4().hex,
            original_url=self._original_url,
            canonical_url=self._canonical_url,
        )
        attempt._append_event("start")  # Record start eagerly
        self._attempt = attempt
        return attempt

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # pragma: no cover - trivial
        if self._attempt is None:
            return
        if exc is not None and not self._attempt._completed:
            self._attempt.record_error(
                error=str(exc),
                error_type=exc_type.__name__ if exc_type else None,
            )
            self._attempt.complete("error")
        self._attempt.ensure_completed()


<<<<<<< ours
__all__ = ["TelemetryWayback", "TelemetryWaybackAttempt"]
=======
"""Telemetry helpers for the Wayback fallback pipeline.

This module provides a very small abstraction that mimics the portions of
the internal telemetry surface used by the tests in this kata.  The real
project tracks a significant amount of metadata; however, for the purposes of
these exercises we only need deterministic counters for the number of
candidate and discovery events emitted during an attempt, plus a lightweight
CDX sampling budget.  Historically the counters lived on the telemetry
instance which meant that successive attempts accidentally shared state.  The
tests target a regression where a single run produced multiple attempts and
the counters were never reset, resulting in incorrect sampling decisions.

The implementation below stores the counters directly on the
``AttemptContext`` returned by :func:`WaybackTelemetry.start_attempt`.  Each
call now receives a fresh context with zeroed counters and a replenished
sampling budget.  The emitters simply consult and update the context so the
behaviour is scoped to a single attempt.
"""

from dataclasses import dataclass, field


@dataclass
class AttemptContext:
    """Mutable state used while recording telemetry for a single attempt.

    Parameters
    ----------
    attempt_id:
        Arbitrary identifier for logging.  The value is not interpreted by the
        module but helps consumers correlate events if desired.
    cdx_sample_budget:
        Optional number of CDX events (candidate or discovery) that should be
        sampled for richer logging.  ``None`` disables sampling limits.
    metadata:
        Free-form mapping persisted across emissions.  Tests use this to carry
        through extra attributes without worrying about additional state.
    """

    attempt_id: str | None
    cdx_sample_budget: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    discovery_count: int = 0
    _cdx_sample_used: int = 0

    def _consume_sample_slot(self) -> bool:
        """Return ``True`` if the current emission should be sampled.

        When ``cdx_sample_budget`` is ``None`` the caller receives ``True`` for
        every event.  Otherwise the method tracks how many events have been
        sampled during the attempt and caps the value at the configured budget.
        """

        if self.cdx_sample_budget is None:
            return True

        if self._cdx_sample_used < self.cdx_sample_budget:
            self._cdx_sample_used += 1
            return True

        return False

    def next_candidate(self) -> dict[str, Any]:
        """Return telemetry payload for the next candidate emission."""

        self.candidate_count += 1
        return {
            "attempt_id": self.attempt_id,
            "kind": "candidate",
            "sequence": self.candidate_count,
            "is_cdx_sample": self._consume_sample_slot(),
            **self.metadata,
        }

    def next_discovery(self) -> dict[str, Any]:
        """Return telemetry payload for the next discovery emission."""

        self.discovery_count += 1
        return {
            "attempt_id": self.attempt_id,
            "kind": "discovery",
            "sequence": self.discovery_count,
            "is_cdx_sample": self._consume_sample_slot(),
            **self.metadata,
        }


class WaybackTelemetry:
    """Tiny façade mirroring the behaviour exercised by the tests."""

    def __init__(self, *, cdx_sample_budget: int | None = None) -> None:
        self._cdx_sample_budget = cdx_sample_budget

    def start_attempt(
        self,
        attempt_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AttemptContext:
        """Start a new telemetry attempt.

        Each invocation returns a freshly-initialised :class:`AttemptContext`
        whose counters begin at zero and whose sampling budget is reset to the
        configured limit.  The returned context is safe to mutate by callers.
        """

        return AttemptContext(
            attempt_id=attempt_id,
            cdx_sample_budget=self._cdx_sample_budget,
            metadata=dict(metadata or {}),
        )

    def emit_candidate(self, context: AttemptContext, **extra: Any) -> dict[str, Any]:
        """Emit a candidate event derived from ``context``.

        The payload merges the automatically managed counters with any caller
        provided ``extra`` keyword arguments.
        """

        payload = context.next_candidate()
        if extra:
            payload.update(extra)
        return payload

    def emit_discovery(self, context: AttemptContext, **extra: Any) -> dict[str, Any]:
        """Emit a discovery event derived from ``context``."""

        payload = context.next_discovery()
        if extra:
            payload.update(extra)
        return payload


__all__ = ["AttemptContext", "WaybackTelemetry"]
>>>>>>> theirs
