"""Tests for observability event system.

Covers:
- Event creation and validation
- Context management (set/get/clear)
- Event serialization (to_dict, to_json)
- Sink registration and emission
- Error handling
"""

import json
from unittest.mock import Mock

import pytest

from DocsToKG.OntologyDownload.observability.events import (
    Event,
    EventContext,
    EventIds,
    clear_context,
    emit_event,
    get_context,
    register_sink,
    set_context,
)


class TestEventModel:
    """Test Event dataclass."""

    def test_event_creation_minimal(self):
        """Event can be created with minimal fields."""
        ctx = EventContext(
            app_version="1.0.0",
            os_name="Linux",
            python_version="3.10.0",
        )
        event = Event(
            ts="2025-10-20T01:23:45.678Z",
            type="test.event",
            level="INFO",
            run_id="run-123",
            config_hash="hash-abc",
            service="test",
            context=ctx,
            ids=EventIds(),
            payload={"key": "value"},
        )
        assert event.type == "test.event"
        assert event.level == "INFO"
        assert event.run_id == "run-123"

    def test_event_to_dict(self):
        """Event.to_dict() converts to proper dict."""
        ctx = EventContext(
            app_version="1.0.0",
            os_name="Linux",
            python_version="3.10.0",
        )
        event = Event(
            ts="2025-10-20T01:23:45Z",
            type="test.event",
            level="WARN",
            run_id="run-123",
            config_hash="hash-abc",
            service="ols",
            context=ctx,
            ids=EventIds(request_id="req-xyz"),
            payload={"entries": 100},
        )
        d = event.to_dict()
        assert d["ts"] == "2025-10-20T01:23:45Z"
        assert d["type"] == "test.event"
        assert d["level"] == "WARN"
        assert d["payload"]["entries"] == 100
        assert d["ids"]["request_id"] == "req-xyz"

    def test_event_to_json(self):
        """Event.to_json() produces valid JSON."""
        ctx = EventContext(
            app_version="1.0.0",
            os_name="Linux",
            python_version="3.10.0",
        )
        event = Event(
            ts="2025-10-20T01:23:45Z",
            type="test.event",
            level="INFO",
            run_id="run-123",
            config_hash="hash-abc",
            service="test",
            context=ctx,
            ids=EventIds(),
            payload={"data": "value"},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "test.event"
        assert parsed["payload"]["data"] == "value"

    def test_event_ids_optional(self):
        """EventIds fields are optional."""
        ids = EventIds()
        assert ids.version_id is None
        assert ids.artifact_id is None
        assert ids.file_id is None
        assert ids.request_id is None

    def test_event_ids_with_values(self):
        """EventIds can hold values."""
        ids = EventIds(
            version_id="v123",
            artifact_id="a456",
            file_id="f789",
            request_id="r999",
        )
        assert ids.version_id == "v123"
        assert ids.artifact_id == "a456"
        assert ids.file_id == "f789"
        assert ids.request_id == "r999"

    def test_event_context_required_fields(self):
        """EventContext requires app_version, os_name, python_version."""
        ctx = EventContext(
            app_version="1.0.0",
            os_name="Linux",
            python_version="3.10.0",
        )
        assert ctx.app_version == "1.0.0"
        assert ctx.os_name == "Linux"
        assert ctx.python_version == "3.10.0"

    def test_event_context_optional_fields(self):
        """EventContext has optional fields."""
        ctx = EventContext(
            app_version="1.0.0",
            os_name="Linux",
            python_version="3.10.0",
            libarchive_version="3.6.0",
            hostname="myhost",
            pid=12345,
        )
        assert ctx.libarchive_version == "3.6.0"
        assert ctx.hostname == "myhost"
        assert ctx.pid == 12345


class TestContextManagement:
    """Test context management functions."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def teardown_method(self):
        """Clean up after each test."""
        clear_context()

    def test_set_context_all_fields(self):
        """set_context() sets all fields."""
        set_context(
            run_id="run-123",
            config_hash="hash-abc",
            service="ols",
        )
        ctx = get_context()
        assert ctx["run_id"] == "run-123"
        assert ctx["config_hash"] == "hash-abc"
        assert ctx["service"] == "ols"

    def test_set_context_partial(self):
        """set_context() can set individual fields."""
        set_context(run_id="run-123")
        ctx = get_context()
        assert ctx["run_id"] == "run-123"
        assert ctx["config_hash"] is None
        assert ctx["service"] is None

    def test_set_context_override(self):
        """set_context() can override previous values."""
        set_context(run_id="run-1")
        set_context(run_id="run-2")
        ctx = get_context()
        assert ctx["run_id"] == "run-2"

    def test_get_context_empty(self):
        """get_context() returns dict with None values initially."""
        ctx = get_context()
        assert ctx["run_id"] is None
        assert ctx["config_hash"] is None
        assert ctx["service"] is None

    def test_clear_context(self):
        """clear_context() resets all values to None."""
        set_context(
            run_id="run-123",
            config_hash="hash-abc",
            service="ols",
        )
        clear_context()
        ctx = get_context()
        assert ctx["run_id"] is None
        assert ctx["config_hash"] is None
        assert ctx["service"] is None

    def test_clear_context_idempotent(self):
        """clear_context() is idempotent."""
        clear_context()
        clear_context()
        clear_context()
        ctx = get_context()
        assert all(v is None for v in ctx.values())


class TestEventEmission:
    """Test emit_event() function."""

    def setup_method(self):
        """Clear context and reset sinks before each test."""
        clear_context()
        # Reset the sinks list (hack: reimport module)
        import DocsToKG.OntologyDownload.observability.events as events_mod

        events_mod._sinks.clear()

    def test_emit_event_minimal(self):
        """emit_event() can be called with just type."""
        event = emit_event(type="test.event")
        assert event.type == "test.event"
        assert event.level == "INFO"
        assert event.run_id is not None  # Auto-generated UUID
        assert event.config_hash == "unknown"
        assert event.service == "default"

    def test_emit_event_with_payload(self):
        """emit_event() includes payload."""
        payload = {"key": "value", "count": 42}
        event = emit_event(type="test.event", payload=payload)
        assert event.payload == payload

    def test_emit_event_with_level(self):
        """emit_event() respects level parameter."""
        event = emit_event(type="test.event", level="ERROR")
        assert event.level == "ERROR"

    def test_emit_event_uses_context(self):
        """emit_event() uses context values if not overridden."""
        set_context(run_id="run-context", config_hash="hash-context", service="ols")
        event = emit_event(type="test.event")
        assert event.run_id == "run-context"
        assert event.config_hash == "hash-context"
        assert event.service == "ols"

    def test_emit_event_overrides_context(self):
        """emit_event() can override context values."""
        set_context(run_id="run-context")
        event = emit_event(type="test.event", run_id="run-override")
        assert event.run_id == "run-override"

    def test_emit_event_with_ids(self):
        """emit_event() includes correlation IDs."""
        event = emit_event(
            type="test.event",
            version_id="v123",
            artifact_id="a456",
            file_id="f789",
            request_id="r999",
        )
        assert event.ids.version_id == "v123"
        assert event.ids.artifact_id == "a456"
        assert event.ids.file_id == "f789"
        assert event.ids.request_id == "r999"

    def test_emit_event_invalid_level(self):
        """emit_event() raises on invalid level."""
        with pytest.raises(ValueError, match="Invalid level"):
            emit_event(type="test.event", level="INVALID")

    def test_emit_event_missing_type(self):
        """emit_event() raises if type is missing."""
        with pytest.raises(ValueError, match="type and level are required"):
            emit_event(type="", level="INFO")

    def test_emit_event_valid_levels(self):
        """emit_event() accepts valid levels."""
        for level in ("INFO", "WARN", "ERROR"):
            event = emit_event(type="test.event", level=level)
            assert event.level == level


class TestSinkRegistration:
    """Test sink registration and emission."""

    def setup_method(self):
        """Clear sinks before each test."""
        import DocsToKG.OntologyDownload.observability.events as events_mod

        events_mod._sinks.clear()

    def test_register_sink(self):
        """register_sink() adds a sink."""
        mock_sink = Mock()
        mock_sink.emit = Mock()
        register_sink(mock_sink)
        emit_event(type="test.event")
        # Sink's emit should have been called
        mock_sink.emit.assert_called_once()

    def test_multiple_sinks(self):
        """Multiple sinks receive events."""
        sink1 = Mock()
        sink1.emit = Mock()
        sink2 = Mock()
        sink2.emit = Mock()
        register_sink(sink1)
        register_sink(sink2)
        emit_event(type="test.event")
        sink1.emit.assert_called_once()
        sink2.emit.assert_called_once()

    def test_sink_receives_event_object(self):
        """Sink receives the Event object."""
        captured_event = None

        def capture_event(event):
            nonlocal captured_event
            captured_event = event

        sink = Mock()
        sink.emit = capture_event
        register_sink(sink)
        emit_event(type="test.event", payload={"data": "test"})
        assert captured_event is not None
        assert captured_event.type == "test.event"
        assert captured_event.payload["data"] == "test"

    def test_sink_error_doesnt_break_emission(self):
        """Sink error doesn't break emit_event()."""
        bad_sink = Mock()
        bad_sink.emit = Mock(side_effect=Exception("Sink error"))
        good_sink = Mock()
        good_sink.emit = Mock()
        register_sink(bad_sink)
        register_sink(good_sink)
        # Should not raise
        event = emit_event(type="test.event")
        assert event is not None
        good_sink.emit.assert_called_once()
