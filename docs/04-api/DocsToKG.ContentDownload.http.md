# 1. Module: http

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.http``.

HTTP Utilities with Backoff Support

This module consolidates resilient HTTP helper functions used across the content
download pipeline. It provides helpers for parsing retry metadata and executing
requests with exponential backoff, jitter, and respect for origin-provided
``Retry-After`` headers.

Key Features:
- Parsing of both integer and HTTP-date ``Retry-After`` formats.
- Unified retry loop with jitter, backoff, and status-code filters.
- Logging hooks for visibility into retry behaviour.

Dependencies:
- `requests`: Primary HTTP client used by resolver sessions.
- `datetime`: Used to interpret HTTP-date headers.

Usage:
    from DocsToKG.ContentDownload.http import request_with_retries

    session = requests.Session()
    response = request_with_retries(session, "GET", "https://example.org/resource")

## 1. Functions

### `parse_retry_after_header(response)`

Parse ``Retry-After`` header and return wait time in seconds.

Args:
response: HTTP response potentially containing ``Retry-After`` header.

Returns:
Float seconds to wait, or ``None`` if header missing/invalid.

Raises:
None: Invalid headers are tolerated and yield ``None`` without raising.

Examples:
>>> # Integer format
>>> response = requests.Response()
>>> response.headers = {"Retry-After": "5"}
>>> parse_retry_after_header(response)
5.0

>>> # HTTP-date format
>>> response.headers = {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}
>>> isinstance(parse_retry_after_header(response), float)
True

### `request_with_retries(session, method, url)`

Execute an HTTP request with exponential backoff and retry handling.

Args:
session: :class:`requests.Session` used to execute the request.
method: HTTP method such as ``"GET"`` or ``"HEAD"``.
url: Fully-qualified URL for the request.
max_retries: Maximum number of retry attempts before returning the final
response or raising an exception. Defaults to ``3``.
retry_statuses: Optional set of HTTP status codes that should trigger a
retry. Defaults to ``{429, 500, 502, 503, 504}``.
backoff_factor: Base multiplier for exponential backoff delays.
Defaults to ``0.75`` seconds.
respect_retry_after: Whether to parse and obey ``Retry-After`` headers
when provided by the server. Defaults to ``True``.
**kwargs: Additional keyword arguments forwarded directly to
:meth:`requests.Session.request`.

Returns:
A :class:`requests.Response` instance on success. The caller is
responsible for closing the response when streaming content.

Raises:
ValueError: If ``max_retries`` or ``backoff_factor`` are invalid or
``url``/``method`` are empty.
requests.RequestException: If all retry attempts fail due to network
errors or the session raises an exception.

Example:
>>> session = requests.Session()
>>> response = request_with_retries(session, "HEAD", "https://example.org")
>>> response.status_code  # doctest: +SKIP
200
>>> with request_with_retries(
...     session,
...     "GET",
...     "https://example.org/resource",
...     stream=True,
... ) as resp:  # doctest: +SKIP
...     data = resp.content

Thread Safety:
The helper is thread-safe provided the supplied ``session`` can be used
safely across threads. Standard :class:`requests.Session` instances are
generally safe for concurrent reads when configured with connection
pooling adapters.

### `request_func()`

Invoke the fallback HTTP method when ``Session.request`` is unavailable.

Args:
method: HTTP method name passed through for parity with ``Session.request``.
url: URL targeted by the outgoing request.
**call_kwargs: Keyword arguments forwarded to the fallback method.

Returns:
``requests.Response`` produced by the fallback invocation.
