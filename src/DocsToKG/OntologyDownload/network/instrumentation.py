# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.network.instrumentation",
#   "purpose": "HTTP network layer instrumentation and telemetry.",
#   "sections": [
#     {
#       "id": "create-http-event-hooks",
#       "name": "create_http_event_hooks",
#       "anchor": "function-create-http-event-hooks",
#       "kind": "function"
#     },
#     {
#       "id": "redact-url",
#       "name": "_redact_url",
#       "anchor": "function-redact-url",
#       "kind": "function"
#     },
#     {
#       "id": "get-attempt-number",
#       "name": "_get_attempt_number",
#       "anchor": "function-get-attempt-number",
#       "kind": "function"
#     },
#     {
#       "id": "is-reused-connection",
#       "name": "_is_reused_connection",
#       "anchor": "function-is-reused-connection",
#       "kind": "function"
#     },
#     {
#       "id": "get-cache-state",
#       "name": "_get_cache_state",
#       "anchor": "function-get-cache-state",
#       "kind": "function"
#     },
#     {
#       "id": "get-ttfb-ms",
#       "name": "_get_ttfb_ms",
#       "anchor": "function-get-ttfb-ms",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""HTTP network layer instrumentation and telemetry.

Emits net.request events for all HTTP calls made by the HTTPX client,
capturing method, URL, status, timings, and cache state.
"""

import time
from typing import Any

from DocsToKG.OntologyDownload.observability.events import emit_event


def create_http_event_hooks() -> dict:
    """Create HTTPX event hooks for telemetry emission.

    Returns:
        Dict with 'request' and 'response' hooks for HTTPX client

    Usage:
        >>> import httpx
        >>> hooks = create_http_event_hooks()
        >>> client = httpx.Client(event_hooks=hooks)
    """
    request_start_time: dict[int, float] = {}

    def on_request(request: Any) -> None:
        """Called when request starts."""
        request_id = id(request)
        request_start_time[request_id] = time.perf_counter()

    def on_response(response: Any) -> None:
        """Called when response completes."""
        request_id = id(response.request)
        start_time = request_start_time.pop(request_id, None)

        if start_time is None:
            return

        elapsed_sec = time.perf_counter() - start_time
        elapsed_ms = elapsed_sec * 1000

        try:
            # Extract URL, redacting sensitive query params
            url = str(response.request.url)
            url_redacted = _redact_url(url)

            # Emit telemetry event
            emit_event(
                type="net.request",
                level="INFO",
                payload={
                    "method": response.request.method,
                    "url_redacted": url_redacted,
                    "host": response.request.url.host or "unknown",
                    "status": response.status_code,
                    "attempt": _get_attempt_number(response),
                    "http2": response.http_version.startswith("HTTP/2"),
                    "reused_conn": _is_reused_connection(response),
                    "cache": _get_cache_state(response),
                    "ttfb_ms": _get_ttfb_ms(response),
                    "elapsed_ms": elapsed_ms,
                    "bytes_read": len(response.content),
                },
            )
        except Exception:
            # Never fail telemetry
            pass

    return {
        "request": [on_request],
        "response": [on_response],
    }


def _redact_url(url: str) -> str:
    """Redact sensitive query parameters from URL.

    Strips query strings by default, keeping only scheme + host + path.
    """
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        # Keep only scheme, netloc, path (no query, fragment, etc.)
        redacted = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        return redacted
    except Exception:
        return "[URL_REDACTION_FAILED]"


def _get_attempt_number(response: Any) -> int:
    """Extract attempt number from response history if available."""
    # HTTPX doesn't track attempt in response, so default to 1
    return 1


def _is_reused_connection(response: Any) -> bool:
    """Detect if connection was reused (heuristic: not first request)."""
    # HTTPX doesn't expose connection reuse info directly
    # We'd need to hook into connection pool to get this
    return False


def _get_cache_state(response: Any) -> str:
    """Get cache state from response extensions."""
    try:
        extensions = response.extensions or {}
        cache_status = extensions.get("cache", {}).get("status", "unknown")
        return cache_status
    except Exception:
        return "unknown"


def _get_ttfb_ms(response: Any) -> float | None:
    """Get time-to-first-byte if available."""
    # HTTPX doesn't track TTFB by default
    return None


__all__ = [
    "create_http_event_hooks",
]
