"""Network subsystem: HTTP client, caching, rate-limiting, and retry policies.

This package provides a unified, production-ready HTTP client stack based on:
- HTTPX: HTTP/1.1 and HTTP/2 client with connection pooling
- Hishel: RFC 9111-compliant HTTP caching
- Tenacity: Robust retry policies with exponential backoff
- pyrate-limiter: Multi-window rate-limiting with cross-process support

Example:
    >>> from DocsToKG.OntologyDownload.network import get_http_client
    >>> client = get_http_client()
    >>> response = client.get("https://api.example.com/data")
"""

from DocsToKG.OntologyDownload.network.client import (
    close_http_client,
    get_http_client,
    reset_http_client,
)
from DocsToKG.OntologyDownload.network.policy import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_WRITE_TIMEOUT,
    KEEPALIVE_EXPIRY,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
)

__all__ = [
    "get_http_client",
    "close_http_client",
    "reset_http_client",
    "HTTP_CONNECT_TIMEOUT",
    "HTTP_READ_TIMEOUT",
    "HTTP_WRITE_TIMEOUT",
    "HTTP_POOL_TIMEOUT",
    "MAX_CONNECTIONS",
    "MAX_KEEPALIVE_CONNECTIONS",
    "KEEPALIVE_EXPIRY",
]
