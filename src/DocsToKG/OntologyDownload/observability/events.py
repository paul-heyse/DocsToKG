# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.observability.events",
#   "purpose": "Canonical Event model and emission system.",
#   "sections": [
#     {
#       "id": "eventids",
#       "name": "EventIds",
#       "anchor": "class-eventids",
#       "kind": "class"
#     },
#     {
#       "id": "eventcontext",
#       "name": "EventContext",
#       "anchor": "class-eventcontext",
#       "kind": "class"
#     },
#     {
#       "id": "event",
#       "name": "Event",
#       "anchor": "class-event",
#       "kind": "class"
#     },
#     {
#       "id": "set-context",
#       "name": "set_context",
#       "anchor": "function-set-context",
#       "kind": "function"
#     },
#     {
#       "id": "get-context",
#       "name": "get_context",
#       "anchor": "function-get-context",
#       "kind": "function"
#     },
#     {
#       "id": "clear-context",
#       "name": "clear_context",
#       "anchor": "function-clear-context",
#       "kind": "function"
#     },
#     {
#       "id": "flush-events",
#       "name": "flush_events",
#       "anchor": "function-flush-events",
#       "kind": "function"
#     },
#     {
#       "id": "register-sink",
#       "name": "register_sink",
#       "anchor": "function-register-sink",
#       "kind": "function"
#     },
#     {
#       "id": "emit-event",
#       "name": "emit_event",
#       "anchor": "function-emit-event",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Canonical Event model and emission system.

Defines the Event dataclass (present on every emitted event), context management
for correlated runs, and the emit_event() function that writes to registered sinks.

Event envelope (present on ALL events):
  - ts: UTC ISO 8601 timestamp
  - type: namespaced event type (e.g., "net.request", "extract.done")
  - level: INFO|WARN|ERROR
  - run_id: correlates CLI command with all nested events
  - config_hash: hash of normalized settings
  - service: service name (e.g., "ols", "bioportal")
  - context: {app_version, os, python, libarchive_version}
  - ids: {version_id, artifact_id, file_id, request_id}
  - payload: event-specific fields (varies by type)
"""

import contextvars
import json
import logging
import os
import platform
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Context Variables (for correlation)
# ============================================================================

_run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
_config_hash_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "config_hash", default=None
)
_service_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("service", default=None)


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class EventIds:
    """IDs for correlating events across subsystems."""

    version_id: str | None = None
    artifact_id: str | None = None
    file_id: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class EventContext:
    """Runtime context captured at event emission."""

    app_version: str
    os_name: str
    python_version: str
    libarchive_version: str | None = None
    hostname: str | None = None
    pid: int | None = None


@dataclass(frozen=True)
class Event:
    """Canonical event envelope (present on every event).

    All fields are required at emission; the schema is immutable and
    can be validated against JSON Schema for compliance.
    """

    # Timestamp & type
    ts: str  # ISO 8601 UTC
    type: str  # namespaced: "net.request", "extract.done", etc.
    level: str  # INFO|WARN|ERROR

    # Correlation & provenance
    run_id: str  # UUID correlating entire CLI command
    config_hash: str  # Hash of normalized settings
    service: str  # Service name (e.g., "ols")

    # Context
    context: EventContext

    # IDs for cross-subsystem correlation
    ids: EventIds

    # Event-specific payload
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "ts": self.ts,
            "type": self.type,
            "level": self.level,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "service": self.service,
            "context": asdict(self.context),
            "ids": asdict(self.ids),
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# ============================================================================
# Context Management
# ============================================================================


def set_context(
    run_id: str | None = None,
    config_hash: str | None = None,
    service: str | None = None,
) -> None:
    """Set correlation context for emitted events.

    Args:
        run_id: UUID for this command run
        config_hash: Hash of normalized settings
        service: Service name (e.g., "ols")

    Example:
        >>> set_context(run_id="uuid-123", config_hash="sha256-abc", service="ols")
    """
    if run_id is not None:
        _run_id_var.set(run_id)
    if config_hash is not None:
        _config_hash_var.set(config_hash)
    if service is not None:
        _service_var.set(service)


def get_context() -> dict[str, str | None]:
    """Get current correlation context.

    Returns:
        Dict with run_id, config_hash, service (may have None values)
    """
    return {
        "run_id": _run_id_var.get(),
        "config_hash": _config_hash_var.get(),
        "service": _service_var.get(),
    }


def clear_context() -> None:
    """Clear correlation context (safe to call; idempotent)."""
    _run_id_var.set(None)
    _config_hash_var.set(None)
    _service_var.set(None)


def flush_events() -> None:
    """Flush all registered event sinks.

    Called at CLI exit and boundary completion to ensure all buffered
    events are persisted to their backends (database, files, etc.).
    """
    for sink in _sinks:
        try:
            if hasattr(sink, "flush"):
                sink.flush()
        except Exception as e:
            logger.error(f"Error flushing sink {sink.__class__.__name__}: {e}")


# ============================================================================
# Event Emission
# ============================================================================

# Pluggable sinks (will be populated by emitters module)
_sinks: list = []


def register_sink(sink) -> None:
    """Register an event sink (internal use; called by emitters module)."""
    _sinks.append(sink)


def emit_event(
    type: str,
    level: str = "INFO",
    payload: dict[str, Any] | None = None,
    run_id: str | None = None,
    config_hash: str | None = None,
    service: str | None = None,
    version_id: str | None = None,
    artifact_id: str | None = None,
    file_id: str | None = None,
    request_id: str | None = None,
) -> Event:
    """Emit a structured event to all registered sinks.

    Args:
        type: Event type (e.g., "net.request", "extract.done")
        level: Event level: INFO|WARN|ERROR
        payload: Event-specific data (dict)
        run_id: Override context run_id
        config_hash: Override context config_hash
        service: Override context service
        version_id: For correlation with version catalog
        artifact_id: For correlation with artifact catalog
        file_id: For correlation with file catalog
        request_id: For correlation with HTTP requests

    Returns:
        The emitted Event

    Raises:
        ValueError: If required fields are missing
    """
    # Use context or override
    ctx = get_context()
    run_id = run_id or ctx["run_id"] or str(uuid.uuid4())
    config_hash = config_hash or ctx["config_hash"] or "unknown"
    service = service or ctx["service"] or "default"

    # Validate required fields
    if not type or not level:
        raise ValueError("Event type and level are required")
    if level not in ("INFO", "WARN", "ERROR"):
        raise ValueError(f"Invalid level: {level}; must be INFO|WARN|ERROR")

    # Build event
    event = Event(
        ts=datetime.now(UTC).isoformat(),
        type=type,
        level=level,
        run_id=run_id,
        config_hash=config_hash,
        service=service,
        context=EventContext(
            app_version="1.0.0",  # TODO: get from settings
            os_name=platform.system(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            libarchive_version=None,  # TODO: try to import libarchive version
            hostname=os.getenv("HOSTNAME", None),
            pid=os.getpid(),
        ),
        ids=EventIds(
            version_id=version_id,
            artifact_id=artifact_id,
            file_id=file_id,
            request_id=request_id,
        ),
        payload=payload or {},
    )

    # Emit to all registered sinks
    for sink in _sinks:
        try:
            sink.emit(event)
        except Exception as e:
            logger.error(f"Error emitting to sink {sink.__class__.__name__}: {e}")

    return event


__all__ = [
    "Event",
    "EventIds",
    "EventContext",
    "emit_event",
    "set_context",
    "get_context",
    "clear_context",
    "flush_events",
    "register_sink",
]
