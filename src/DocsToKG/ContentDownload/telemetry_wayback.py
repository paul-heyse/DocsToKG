"""Wayback-specific telemetry helpers.

This module provides a minimal telemetry surface used by the Wayback resolver
pipeline.  The :class:`TelemetryWayback` helper offers an opt-in mechanism for
measuring resolver attempts with a monotonic clock.  Tests can inject a fake
monotonic function to obtain deterministic timings which is handy for flaky
scenarios and for validating retry logic.

Two small abstractions power the implementation:

``TelemetryWayback``
    Factory/registry that hands out :class:`AttemptContext` instances and keeps
    the recorded measurements.

``AttemptContext``
    Context manager that captures a monotonic start timestamp when created and
    emits an immutable :class:`AttemptMeasurement` once finished.

Both abstractions accept an optional ``monotonic_fn`` so durations can be
measured using an injected fake clock instead of ``time.monotonic``.  The fake
clock capability is vital for tests that need to advance time deterministically
without sleeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import time

MonotonicFn = Callable[[], float]


def _diff_ms(start: float, end: float) -> int:
    """Return the difference between ``end`` and ``start`` in milliseconds."""

    return int(round((end - start) * 1000))


@dataclass(frozen=True)
class AttemptMeasurement:
    """Measurement captured for a Wayback resolver attempt."""

    attempt_id: str
    start_monotonic: float
    end_monotonic: float
    elapsed_ms: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AttemptContext:
    """Context manager that measures a single resolver attempt.

    The context captures a monotonic start timestamp on construction and stores
    a monotonic end timestamp when :meth:`finish` (or ``__exit__``) is invoked.
    Durations always come from the injected monotonic function, making the
    helper deterministic during unit tests.
    """

    __slots__ = (
        "_telemetry",
        "_metadata",
        "_monotonic_fn",
        "_finished",
        "_measurement",
        "attempt_id",
        "started_monotonic",
        "ended_monotonic",
        "elapsed_ms",
    )

    def __init__(
        self,
        telemetry: "TelemetryWayback",
        *,
        attempt_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._telemetry = telemetry
        self._metadata: Dict[str, Any] = dict(metadata or {})
        self._monotonic_fn: MonotonicFn = telemetry.monotonic_fn
        self._finished = False
        self._measurement: Optional[AttemptMeasurement] = None
        self.attempt_id = attempt_id
        self.started_monotonic = self._monotonic_fn()
        self.ended_monotonic: Optional[float] = None
        self.elapsed_ms: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "AttemptContext":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.finish()
        return False

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------
    def finish(self, **extra_metadata: Any) -> AttemptMeasurement:
        """Finalize the context and return the recorded measurement.

        Repeated invocations simply return the cached measurement.  The
        timestamp and duration values always come from ``self._monotonic_fn``.
        """

        if not self._finished:
            self._finished = True
            self._metadata.update(extra_metadata)
            self.ended_monotonic = self._monotonic_fn()
            end = self.ended_monotonic
            start = self.started_monotonic
            if end is None:  # pragma: no cover - defensive
                raise RuntimeError("Ended monotonic timestamp missing")
            self.elapsed_ms = _diff_ms(start, end)
            measurement = AttemptMeasurement(
                attempt_id=self.attempt_id,
                start_monotonic=start,
                end_monotonic=end,
                elapsed_ms=self.elapsed_ms,
                metadata=dict(self._metadata),
            )
            self._telemetry._record_attempt(measurement)
            self._measurement = measurement
        assert self._measurement is not None  # For type-checkers
        return self._measurement

    def elapsed_ms_so_far(self) -> int:
        """Return the elapsed time in milliseconds without closing the context."""

        current = self._monotonic_fn()
        return _diff_ms(self.started_monotonic, current)


class TelemetryWayback:
    """Collect and expose Wayback resolver telemetry measurements."""

    def __init__(self, *, monotonic_fn: Optional[MonotonicFn] = None) -> None:
        self._monotonic_fn: MonotonicFn = monotonic_fn or time.monotonic
        self._attempts: List[AttemptMeasurement] = []

    # ------------------------------------------------------------------
    # Factories & accessors
    # ------------------------------------------------------------------
    @property
    def monotonic_fn(self) -> MonotonicFn:
        return self._monotonic_fn

    @property
    def attempts(self) -> List[AttemptMeasurement]:
        return self._attempts

    def attempt(self, attempt_id: str, **metadata: Any) -> AttemptContext:
        """Return a new :class:`AttemptContext` bound to this telemetry helper."""

        return AttemptContext(self, attempt_id=attempt_id, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_attempt(self, measurement: AttemptMeasurement) -> None:
        self._attempts.append(measurement)


__all__ = [
    "AttemptContext",
    "AttemptMeasurement",
    "TelemetryWayback",
]

