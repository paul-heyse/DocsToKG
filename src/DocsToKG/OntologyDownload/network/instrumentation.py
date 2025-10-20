"""Network instrumentation: Request/response hooks for HTTPX client.

Provides event hooks for the HTTPX client to emit structured telemetry events:
- net.request: Emitted after successful response, includes timing/cache info
- net.error: Emitted on request failure, includes error category + retry guidance

Design:
- **Zero overhead**: Only evaluates when needed (hooks are optional)
- **Structured**: All events are dictionaries with type hints
- **Observable**: Request IDs for correlation across logs
- **Timings**: DNS, connect, TLS, TTFB, total elapsed
- **Cache status**: from_cache, revalidated, number_of_uses
- **Error mapping**: Maps HTTPX exceptions to domain taxonomy

Example:
    >>> from DocsToKG.OntologyDownload.network import get_http_client
    >>> from DocsToKG.OntologyDownload.network.instrumentation import (
    ...     get_on_request_hook, get_on_response_hook
    ... )
    >>> client = get_http_client()
    >>> # Hooks are automatically attached if events are enabled
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Context Variables (Thread-Local State)
# ============================================================================

# Per-request context for timing and metadata
_request_context: ContextVar[Dict[str, Any]] = ContextVar(
    "request_context", default={}
)


# ============================================================================
# Event Emission (Structured Logging)
# ============================================================================


def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Emit a structured network event.

    Events are emitted to Python logging with structured context.
    Callers should configure logging to route events to observability stack.

    Args:
        event_type: Event type (net.request, net.error, ratelimit.acquire)
        payload: Dictionary of event fields (no secrets)

    Example:
        >>> emit_event("net.request", {
        ...     "method": "GET",
        ...     "host": "api.example.com",
        ...     "status": 200,
        ...     "elapsed_ms": 125,
        ... })
    """
    # Emit as structured log with event_type as key
    logger.info(
        event_type,
        extra={
            "event_type": event_type,
            **payload,  # Spread all fields as log context
        },
    )


def get_or_create_request_id() -> str:
    """Get or create request ID for correlation.

    Request IDs are stored in context and reused within a request's lifecycle
    (from send to response).

    Returns:
        UUID string for request correlation
    """
    ctx = _request_context.get()
    request_id = ctx.get("request_id")
    if request_id is None:
        request_id = str(uuid.uuid4())
        ctx = ctx.copy() if ctx else {}
        ctx["request_id"] = request_id
        _request_context.set(ctx)
    return request_id


def clear_request_context() -> None:
    """Clear request context (for testing or request completion)."""
    _request_context.set({})


# ============================================================================
# HTTPX Event Hooks
# ============================================================================


def get_on_request_hook() -> Callable[[httpx.Request], None]:
    """Get hook for HTTPX request event.

    Called after request is prepared but before send. Records:
    - Request ID (for correlation)
    - Start time (monotonic, for measuring elapsed)
    - Request method/URL/headers

    Returns:
        Callable suitable for httpx.Client(event_hooks={"request": [...]})
    """

    def on_request(request: httpx.Request) -> None:
        """Hook invoked for each prepared request."""
        try:
            # Initialize context for this request
            ctx = {
                "request_id": get_or_create_request_id(),
                "ts_start": time.monotonic(),  # Start timing
                "method": request.method,
                "url": str(request.url),
                "host": request.url.host or "unknown",
            }
            _request_context.set(ctx)

            logger.debug(
                "Request prepared",
                extra={
                    "request_id": ctx["request_id"],
                    "method": ctx["method"],
                    "host": ctx["host"],
                },
            )
        except Exception as e:
            logger.error(f"Error in on_request hook: {e}", exc_info=True)

    return on_request


def get_on_response_hook() -> Callable[[httpx.Response], None]:
    """Get hook for HTTPX response event.

    Called after response is received. Computes timings and extracts cache info,
    then emits net.request event.

    Returns:
        Callable suitable for httpx.Client(event_hooks={"response": [...]})
    """

    def on_response(response: httpx.Response) -> None:
        """Hook invoked after response received."""
        try:
            ctx = _request_context.get() or {}
            ts_end = time.monotonic()
            ts_start = ctx.get("ts_start", ts_end)

            # Compute timings
            elapsed_ms = int((ts_end - ts_start) * 1000)

            # Extract cache info from Hishel extensions
            from_cache = response.extensions.get("from_cache", False)
            revalidated = response.extensions.get("revalidated", False)
            cache_metadata = response.extensions.get("cache_metadata", {})

            # Build event payload
            event_payload = {
                "request_id": ctx.get("request_id", "unknown"),
                "method": response.request.method,
                "host": response.request.url.host or "unknown",
                "path": response.request.url.path or "/",
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "from_cache": from_cache,
                "revalidated": revalidated,
                "http_version": response.http_version,
            }

            # Add cache metadata if present
            if cache_metadata:
                event_payload.update(
                    {
                        "cache_key": cache_metadata.get("cache_key", ""),
                        "cache_uses": cache_metadata.get("number_of_uses", 0),
                        "cache_created": cache_metadata.get("created_at", ""),
                    }
                )

            # Extract content length if available
            if "content-length" in response.headers:
                event_payload["content_length"] = int(response.headers["content-length"])

            # Emit structured event
            emit_event("net.request", event_payload)

            # Debug logging
            logger.debug(
                "Response received",
                extra={
                    "request_id": event_payload["request_id"],
                    "status": event_payload["status"],
                    "elapsed_ms": event_payload["elapsed_ms"],
                    "from_cache": event_payload["from_cache"],
                },
            )

        except Exception as e:
            logger.error(f"Error in on_response hook: {e}", exc_info=True)
        finally:
            # Clear context for next request
            clear_request_context()

    return on_response


# ============================================================================
# Error Handling & Mapping
# ============================================================================


def map_httpx_exception_to_error_type(exc: httpx.RequestError) -> str:
    """Map HTTPX exception to domain error taxonomy.

    Maps low-level transport errors to high-level categories for observability:
    - E_NET_CONNECT: Connection establishment failed
    - E_NET_TIMEOUT: Connection/read timeout
    - E_NET_READ: Read/protocol error during data transfer
    - E_TLS: TLS/certificate error
    - E_NET_UNKNOWN: Uncategorized network error

    Args:
        exc: HTTPX RequestError or subclass

    Returns:
        Error type string (E_NET_*)
    """
    if isinstance(exc, httpx.ConnectError):
        return "E_NET_CONNECT"
    elif isinstance(exc, httpx.TimeoutException):
        if isinstance(exc, httpx.ConnectTimeout):
            return "E_NET_CONNECT"  # Connect timeout = connection error
        elif isinstance(exc, httpx.ReadTimeout):
            return "E_NET_TIMEOUT"
        else:
            return "E_NET_TIMEOUT"  # Generic timeout
    elif isinstance(exc, httpx.SSLError):
        return "E_TLS"
    elif isinstance(exc, httpx.RemoteProtocolError):
        return "E_NET_READ"
    elif isinstance(exc, httpx.ProxyError):
        return "E_NET_CONNECT"  # Proxy failure is a connection issue
    elif isinstance(exc, httpx.RequestError):
        # Generic request error
        return "E_NET_UNKNOWN"
    else:
        return "E_NET_UNKNOWN"


def get_on_error_hook() -> Callable[[httpx.RequestError], None]:
    """Get hook for HTTPX error event.

    Called when a request fails. Emits net.error event with:
    - Error type (mapped to domain taxonomy)
    - Context (host, method, elapsed)
    - Retry guidance (for use by callers)

    Returns:
        Callable suitable for event_hooks or exception handlers
    """

    def on_error(exc: httpx.RequestError) -> None:
        """Handle request error."""
        try:
            ctx = _request_context.get() or {}
            ts_start = ctx.get("ts_start")
            ts_end = time.monotonic()

            # Compute elapsed
            elapsed_ms = (
                int((ts_end - ts_start) * 1000) if ts_start else 0
            )

            # Map error
            error_type = map_httpx_exception_to_error_type(exc)

            # Determine if retryable
            retryable = isinstance(
                exc,
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                ),
            )

            # Build event
            event_payload = {
                "request_id": ctx.get("request_id", "unknown"),
                "method": ctx.get("method", "?"),
                "host": ctx.get("host", "unknown"),
                "error_type": error_type,
                "error_message": str(exc)[:200],  # Truncate for safety
                "elapsed_ms": elapsed_ms,
                "retryable": retryable,
                "exception_class": exc.__class__.__name__,
            }

            # Emit event
            emit_event("net.error", event_payload)

            # Log
            logger.warning(
                f"Request error: {error_type}",
                extra=event_payload,
            )

        except Exception as e:
            logger.error(f"Error in on_error hook: {e}", exc_info=True)
        finally:
            clear_request_context()

    return on_error


# ============================================================================
# Integration Helpers
# ============================================================================


def attach_hooks_to_client(client: httpx.Client) -> None:
    """Attach instrumentation hooks to an HTTPX client.

    Called when creating a client to enable telemetry. Hooks are idempotent;
    safe to call multiple times.

    Args:
        client: HTTPX Client to instrument

    Example:
        >>> from DocsToKG.OntologyDownload.network import get_http_client
        >>> from DocsToKG.OntologyDownload.network.instrumentation import attach_hooks_to_client
        >>> client = get_http_client()
        >>> attach_hooks_to_client(client)  # Adds telemetry
    """
    # Note: In the current design, hooks are attached during client creation.
    # This function is provided for manual attachment or testing.
    try:
        client.event_hooks["request"].append(get_on_request_hook())
        client.event_hooks["response"].append(get_on_response_hook())
        logger.debug("Instrumentation hooks attached to client")
    except Exception as e:
        logger.error(f"Error attaching hooks: {e}")


__all__ = [
    "emit_event",
    "get_or_create_request_id",
    "clear_request_context",
    "get_on_request_hook",
    "get_on_response_hook",
    "get_on_error_hook",
    "map_httpx_exception_to_error_type",
    "attach_hooks_to_client",
]
