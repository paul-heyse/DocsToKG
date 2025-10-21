# === NAVMAP v1 ===
# {
#   "module": "tests.fixtures.telemetry_fixtures",
#   "purpose": "Telemetry event capture and assertions fixtures",
#   "sections": [
#     {"id": "event-sink-fixture", "name": "event_sink", "anchor": "fixture-event-sink", "kind": "fixture"},
#     {"id": "event-assertions", "name": "EventAssertions", "anchor": "class-event-assertions", "kind": "class"}
#   ]
# }
# === /NAVMAP ===

"""
Telemetry and event capture fixtures for testing observability.

Provides in-memory event sink for capturing and asserting on telemetry events
emitted during test execution. Enables verification of logging, tracing, and
metrics collection without real external systems.
"""

from __future__ import annotations

from typing import Any, Generator

import pytest


class EventAssertions:
    """Helper for asserting on captured telemetry events."""

    def __init__(self, events: list[dict[str, Any]]):
        """Initialize with events list."""
        self.events = events

    def count(self) -> int:
        """Total number of events captured."""
        return len(self.events)

    def by_type(self, event_type: str) -> list[dict[str, Any]]:
        """Filter events by type."""
        return [e for e in self.events if e.get("type") == event_type]

    def by_stage(self, stage: str) -> list[dict[str, Any]]:
        """Filter events by stage."""
        return [e for e in self.events if e.get("stage") == stage]

    def has_event(self, event_type: str) -> bool:
        """Check if event type exists."""
        return any(e.get("type") == event_type for e in self.events)

    def has_error(self) -> bool:
        """Check if any error events were captured."""
        return any(e.get("level") == "error" for e in self.events)

    def filter(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Filter events by arbitrary field values."""
        result = []
        for event in self.events:
            match = True
            for key, value in kwargs.items():
                if event.get(key) != value:
                    match = False
                    break
            if match:
                result.append(event)
        return result

    def assert_count(self, expected: int) -> None:
        """Assert total event count."""
        assert self.count() == expected, f"Expected {expected} events, got {self.count()}"

    def assert_has_type(self, event_type: str) -> None:
        """Assert event type exists."""
        assert self.has_event(event_type), f"No event of type {event_type}"

    def assert_no_errors(self) -> None:
        """Assert no error events were captured."""
        assert not self.has_error(), (
            f"Found error events: {[e for e in self.events if e.get('level') == 'error']}"
        )

    def assert_min_count(self, minimum: int) -> None:
        """Assert minimum event count."""
        assert self.count() >= minimum, f"Expected at least {minimum} events, got {self.count()}"

    def assert_max_count(self, maximum: int) -> None:
        """Assert maximum event count."""
        assert self.count() <= maximum, f"Expected at most {maximum} events, got {self.count()}"


@pytest.fixture
def event_sink() -> Generator[dict[str, Any], None, None]:
    """
    Provide an in-memory telemetry event sink.

    Yields a dict with:
    - events: list of captured events
    - emit: function to emit events
    - capture: context manager for capturing events
    - assertions: EventAssertions helper for assertions
    - reset: clear captured events

    Example:
        def test_with_events(event_sink):
            sink = event_sink
            sink['emit']({'type': 'request', 'method': 'GET', 'status': 200})
            sink['emit']({'type': 'response', 'size_bytes': 1024})

            assert sink['assertions'].count() == 2
            assert sink['assertions'].has_event('request')
            sink['assertions'].assert_no_errors()

            requests = sink['assertions'].by_type('request')
            assert len(requests) == 1
            assert requests[0]['status'] == 200
    """
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        """Emit a telemetry event."""
        # Add metadata
        if "timestamp" not in event:
            import time

            event["timestamp"] = time.time()
        events.append(event)

    def reset() -> None:
        """Clear all captured events."""
        events.clear()

    yield {
        "events": events,
        "emit": emit,
        "assertions": EventAssertions(events),
        "reset": reset,
    }


@pytest.fixture
def ratelimit_registry_reset() -> Generator[None, None, None]:
    """
    Provide rate limiter registry reset for test isolation.

    Automatically resets rate limiter registry before and after test.
    Useful when testing rate limiting with different configurations.

    Example:
        def test_rate_limit_isolation(ratelimit_registry_reset):
            # Rate limiter state is clean
            from DocsToKG.ContentDownload.ratelimit import get_registry
            registry = get_registry()
            assert len(registry.limiters) == 0
            # ... test code ...
            # Registry automatically reset after test
    """
    try:
        from DocsToKG.ContentDownload.ratelimit import get_registry
    except ImportError:
        pytest.skip("Rate limiter not available")

    registry = get_registry()

    # Clear before test
    if hasattr(registry, "limiters"):
        registry.limiters.clear()
    if hasattr(registry, "policies"):
        registry.policies.clear()

    yield

    # Clear after test
    if hasattr(registry, "limiters"):
        registry.limiters.clear()
    if hasattr(registry, "policies"):
        registry.policies.clear()


@pytest.fixture
def mock_event_emitter() -> Generator[dict[str, Any], None, None]:
    """
    Provide a mock event emitter for testing telemetry integration.

    Yields a dict with:
    - emitted: list of all emitted events
    - emit_event: function matching real emitter interface
    - reset: clear emitted events
    - by_type: filter by event type
    - call_count: number of emit_event calls

    Example:
        def test_emitter(mock_event_emitter):
            emitter = mock_event_emitter
            emitter['emit_event']('net.request', {'url': 'https://example.com'})
            emitter['emit_event']('net.response', {'status': 200})

            assert emitter['call_count']() == 2
            assert len(emitter['by_type']('net.request')) == 1
    """
    emitted: list[tuple[str, dict[str, Any]]] = []

    def emit_event(event_type: str, data: dict[str, Any]) -> None:
        """Emit a typed event."""
        emitted.append((event_type, data.copy()))

    def reset() -> None:
        """Clear emitted events."""
        emitted.clear()

    def by_type(event_type: str) -> list[dict[str, Any]]:
        """Get events of specific type."""
        return [data for etype, data in emitted if etype == event_type]

    def call_count() -> int:
        """Get total emit calls."""
        return len(emitted)

    yield {
        "emitted": emitted,
        "emit_event": emit_event,
        "reset": reset,
        "by_type": by_type,
        "call_count": call_count,
    }
