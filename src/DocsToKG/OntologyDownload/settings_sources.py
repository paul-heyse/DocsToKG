"""TracingSettingsSource: Wraps Pydantic SettingsSource to track field attribution.

This module provides utilities to capture which source (CLI, environment, config file, or default)
provided each configuration field. Source attribution is stored in a thread-local context
to support multi-threaded environments.

Example:
    >>> from settings_sources import TracingSettingsSource, get_source_fingerprint
    >>> # Create traced sources
    >>> cli_source = TracingSettingsSource(cli_dict_source, "cli")
    >>> env_source = TracingSettingsSource(env_settings_source, "env")
    >>> # After loading settings:
    >>> fingerprint = get_source_fingerprint()
    >>> # fingerprint = {"http__timeout_connect": "cli", "http__timeout_read": "env", ...}
"""

import contextvars
from typing import Any, Dict, Tuple

from pydantic_settings import PydanticBaseSettingsSource

# Thread-local context to capture source mapping across settings loads
_source_context: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "source_fingerprint", default={}
)


class TracingSettingsSource(PydanticBaseSettingsSource):
    """Wraps a Pydantic SettingsSource to track which source provided each field.

    This wrapper intercepts get_field_value() calls to record source attribution
    in a thread-safe context variable. Sources are never logged with values—
    only source names (cli, env, config:/path, .env.ontofetch, .env, default).

    Attributes:
        source: The underlying SettingsSource to wrap
        source_name: Human-readable name for this source (e.g., "cli", "env", "config:/path/to/file")
    """

    def __init__(self, source: PydanticBaseSettingsSource, source_name: str):
        """Initialize TracingSettingsSource.

        Args:
            source: The underlying SettingsSource to wrap
            source_name: Descriptive name for this source (used in fingerprint)
        """
        self.source = source
        self.source_name = source_name

    def get_field_value(self, field, field_name: str) -> Tuple[Any, str, bool]:
        """Get field value and record source attribution.

        Calls the wrapped source's get_field_value() method and, if a value is found
        (not using default), records the source in the thread-local context.

        Args:
            field: Pydantic FieldInfo object
            field_name: Name of the field being loaded

        Returns:
            Tuple of (value, field_set_name, using_default) from wrapped source
        """
        try:
            value, field_set_name, using_default = self.source.get_field_value(field, field_name)

            # Only record source if we actually got a value (not using default)
            if not using_default:
                ctx = _source_context.get().copy()  # Copy to avoid mutation during iteration
                ctx[field_name] = self.source_name
                _source_context.set(ctx)

            return value, field_set_name, using_default

        except Exception:
            # Re-raise; don't suppress errors in source loading
            raise

    def __call__(self) -> Dict[str, Any]:
        """Fallback for tuple-returning sources.

        Some SettingsSource implementations return a dict directly when called.
        This method provides that interface.

        Returns:
            Dictionary of field values from the wrapped source
        """
        result = self.source()
        if isinstance(result, dict):
            # Record that all returned fields came from this source
            ctx = _source_context.get().copy()
            for field_name in result.keys():
                ctx[field_name] = self.source_name
            _source_context.set(ctx)
        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TracingSettingsSource(source={self.source.__class__.__name__}, name='{self.source_name}')"


def get_source_fingerprint() -> Dict[str, str]:
    """Retrieve accumulated source attribution map.

    This function returns a dictionary mapping field names to source names.
    Sources are: "cli", "env", "config:/path/to/file", ".env.ontofetch", ".env", "default".

    Returns:
        Dictionary where keys are field names and values are source names.
        Empty dict if no sources have been traced yet.

    Example:
        >>> from settings_sources import get_source_fingerprint
        >>> fp = get_source_fingerprint()
        >>> print(fp)
        {'http__timeout_connect': 'cli', 'http__timeout_read': 'env', 'db__path': 'default'}
    """
    return _source_context.get().copy()  # Return copy to prevent external mutation


def set_source_fingerprint(fingerprint: Dict[str, str]) -> None:
    """Set the source fingerprint (primarily for testing).

    Args:
        fingerprint: Dictionary mapping field names to source names
    """
    _source_context.set(fingerprint.copy())


def clear_source_context() -> None:
    """Reset source context (primarily for testing between runs).

    Call this between test cases or before starting a new settings load
    to clear the trace context.
    """
    _source_context.set({})


def init_source_context() -> None:
    """Initialize source context for a new settings load.

    This is typically called automatically by settings loading code,
    but can be explicitly called to reset context at the start of a phase.
    """
    _source_context.set({})


__all__ = [
    "TracingSettingsSource",
    "get_source_fingerprint",
    "set_source_fingerprint",
    "clear_source_context",
    "init_source_context",
]
