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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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

    attempt_id: Optional[str]
    cdx_sample_budget: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
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

    def next_candidate(self) -> Dict[str, Any]:
        """Return telemetry payload for the next candidate emission."""

        self.candidate_count += 1
        return {
            "attempt_id": self.attempt_id,
            "kind": "candidate",
            "sequence": self.candidate_count,
            "is_cdx_sample": self._consume_sample_slot(),
            **self.metadata,
        }

    def next_discovery(self) -> Dict[str, Any]:
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

    def __init__(self, *, cdx_sample_budget: Optional[int] = None) -> None:
        self._cdx_sample_budget = cdx_sample_budget

    def start_attempt(
        self,
        attempt_id: Optional[str] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
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

    def emit_candidate(self, context: AttemptContext, **extra: Any) -> Dict[str, Any]:
        """Emit a candidate event derived from ``context``.

        The payload merges the automatically managed counters with any caller
        provided ``extra`` keyword arguments.
        """

        payload = context.next_candidate()
        if extra:
            payload.update(extra)
        return payload

    def emit_discovery(self, context: AttemptContext, **extra: Any) -> Dict[str, Any]:
        """Emit a discovery event derived from ``context``."""

        payload = context.next_discovery()
        if extra:
            payload.update(extra)
        return payload


__all__ = ["AttemptContext", "WaybackTelemetry"]
