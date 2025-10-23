"""Network subsystem: HTTP client, caching, rate-limiting, and retry policies.

This package provides a unified, production-ready HTTP client stack based on:
- HTTPX: HTTP/1.1 and HTTP/2 client with connection pooling
- Hishel: RFC 9111-compliant HTTP caching
- Tenacity: Robust retry policies with exponential backoff
- pyrate-limiter: Multi-window rate-limiting with cross-process support

Modules:
- client: HTTPX client factory with lazy singleton pattern
- policy: HTTP policy constants (timeouts, pooling, caching)
- instrumentation: Request/response hooks for structured telemetry
- retry: Tenacity-based retry policies for 429/5xx resilience
- redirect: Safe redirect following with security auditing

Example:
    >>> from DocsToKG.OntologyDownload.network import get_http_client
    >>> from DocsToKG.OntologyDownload.network.retry import create_http_retry_policy
    >>> from DocsToKG.OntologyDownload.network.redirect import safe_get_with_redirect
    >>>
    >>> client = get_http_client()
    >>> policy = create_http_retry_policy(max_delay_seconds=30)
    >>>
    >>> for attempt in policy:
    ...     with attempt:
    ...         response, hops = safe_get_with_redirect(client, url)
"""

from DocsToKG.OntologyDownload.network.client import (
    close_http_client,
    get_http_client,
    reset_http_client,
)
from DocsToKG.OntologyDownload.network.instrumentation import create_http_event_hooks
from DocsToKG.OntologyDownload.network.policy import (
    HTTP2_ENABLED,
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_WRITE_TIMEOUT,
    KEEPALIVE_EXPIRY,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
)
from DocsToKG.OntologyDownload.network.polite_client import (
    close_polite_http_client,
    get_polite_http_client,
    reset_polite_http_client,
)
from DocsToKG.OntologyDownload.network.redirect import (
    MaxRedirectsExceeded,
    MissingLocationHeader,
    RedirectError,
    RedirectPolicy,
    UnsafeRedirectTarget,
    format_audit_trail,
    safe_get_with_redirect,
    safe_post_with_redirect,
)
from DocsToKG.OntologyDownload.network.retry import (
    create_aggressive_retry_policy,
    create_http_retry_policy,
    create_idempotent_retry_policy,
    create_rate_limit_retry_policy,
    retry_http_request,
)

__all__ = [
    # Client lifecycle
    "get_http_client",
    "close_http_client",
    "reset_http_client",
    # Polite client (integration)
    "get_polite_http_client",
    "close_polite_http_client",
    "reset_polite_http_client",
    # Timeouts and pooling
    "HTTP_CONNECT_TIMEOUT",
    "HTTP_READ_TIMEOUT",
    "HTTP_WRITE_TIMEOUT",
    "HTTP_POOL_TIMEOUT",
    "MAX_CONNECTIONS",
    "MAX_KEEPALIVE_CONNECTIONS",
    "KEEPALIVE_EXPIRY",
    "HTTP2_ENABLED",
    # Instrumentation
    "create_http_event_hooks",
    # Retry policies
    "create_http_retry_policy",
    "create_idempotent_retry_policy",
    "create_aggressive_retry_policy",
    "create_rate_limit_retry_policy",
    "retry_http_request",
    # Redirect handling
    "safe_get_with_redirect",
    "safe_post_with_redirect",
    "RedirectPolicy",
    "RedirectError",
    "MaxRedirectsExceeded",
    "UnsafeRedirectTarget",
    "MissingLocationHeader",
    "format_audit_trail",
]
