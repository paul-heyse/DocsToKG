"""Wayback resolver telemetry helpers."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


def _ensure_event_buffer(ctx: Any) -> Optional[List[dict[str, Any]]]:
    """Return an event buffer stored on ``ctx`` if telemetry is enabled."""

    if ctx is None:
        return None

    buffer = getattr(ctx, "telemetry_wayback_events", None)
    if buffer is None:
        buffer = []
        try:
            setattr(ctx, "telemetry_wayback_events", buffer)
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
    candidate_url: Optional[str] = None
    events: List[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    _completed: bool = field(default=False, init=False, repr=False)

    def record_discovery(self, metadata: Mapping[str, Any]) -> None:
        """Record discovery metadata for this attempt."""

        self._append_event("discovery", metadata=dict(metadata))

    def record_candidate(
        self,
        outcome: str,
        *,
        candidate_url: Optional[str],
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
        error_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record an error event for the attempt."""

        payload = {"error": error}
        if error_type:
            payload["error_type"] = error_type
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        self._append_event("error", **payload)

    def complete(self, status: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
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
        self._attempt: Optional[TelemetryWaybackAttempt] = None

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

