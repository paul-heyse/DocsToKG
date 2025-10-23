"""Legacy Wayback telemetry façade used by kata fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        """Return ``True`` if the current emission should be sampled."""

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
    """Tiny façade mirroring the behaviour exercised by the kata tests."""

    def __init__(self, *, cdx_sample_budget: int | None = None) -> None:
        self._cdx_sample_budget = cdx_sample_budget

    def start_attempt(
        self,
        attempt_id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AttemptContext:
        """Start a new telemetry attempt and return its context."""

        return AttemptContext(
            attempt_id=attempt_id,
            cdx_sample_budget=self._cdx_sample_budget,
            metadata=dict(metadata or {}),
        )

    def emit_candidate(self, context: AttemptContext, **extra: Any) -> dict[str, Any]:
        """Emit a candidate event derived from ``context``."""

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
