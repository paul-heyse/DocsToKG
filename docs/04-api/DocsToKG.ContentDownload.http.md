# Module: http

Unified HTTP request utilities with retry and backoff support.

## Functions

### `parse_retry_after_header(response)`

Parse ``Retry-After`` header and return wait time in seconds.

Args:
response: HTTP response potentially containing ``Retry-After`` header.

Returns:
Float seconds to wait, or ``None`` if header missing/invalid.

Raises:
None.

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
