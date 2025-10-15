# 1. Module: network

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.network``.

Unified Network Utilities

The network module consolidates session construction, resilient HTTP helpers,
and conditional request tooling that previously lived in separate ``http`` and
``conditional`` modules. Centralising the logic keeps retry defaults, adapter
configuration, and cache semantics aligned across the content download pipeline.

Key Features:
- ``create_session`` factory for pooled ``requests.Session`` instances.
- ``request_with_retries`` helper with exponential backoff and jitter.
- ``ConditionalRequestHelper`` dataclass suite for polite conditional GETs.

Dependencies:
- `requests`: Primary HTTP client used by resolver sessions.
- `datetime`: Used to interpret HTTP-date headers.

Usage:
```python
from DocsToKG.ContentDownload.network import (
    ConditionalRequestHelper,
    create_session,
    request_with_retries,
)

session = create_session({"User-Agent": "DocsToKGDownloader/1.0"})
response = request_with_retries(session, "GET", "https://example.org/resource")
helper = ConditionalRequestHelper(prior_etag="abcd", prior_path="/tmp/file.pdf")
headers = helper.build_headers()
```

## 1. Functions

### `create_session(headers=None, *, adapter_max_retries=0)`

Return a ``requests.Session`` configured for DocsToKG network requests.

Args:
headers: Optional mapping applied to the session header store; the mapping is
copied so caller-owned dictionaries remain untouched.
adapter_max_retries: Retry count configured on mounted adapters. Defaults to
``0`` so :func:`request_with_retries` governs retry behaviour.

Returns:
``requests.Session`` with HTTP/HTTPS adapters mounted for connection pooling.

Notes:
The session is safe to share across threads provided callers avoid mutating
shared state (for example, by modifying ``session.headers`` directly) after
distribution to worker threads.

### `parse_retry_after_header(response)`

Parse ``Retry-After`` header and return wait time in seconds.

Args:
response: HTTP response potentially containing ``Retry-After`` header.

Returns:
Float seconds to wait, or ``None`` if header missing/invalid.

Raises:
None: Invalid headers are tolerated and yield ``None`` without raising.

Examples:
```python
>>> response = requests.Response()
>>> response.headers = {"Retry-After": "5"}
>>> parse_retry_after_header(response)
5.0

>>> response.headers = {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}
>>> isinstance(parse_retry_after_header(response), float)
True
```

### `request_with_retries(session, method, url, **kwargs)`

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
```python
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
```

Thread Safety:
The helper is thread-safe provided the supplied ``session`` can be used
safely across threads. Standard :class:`requests.Session` instances are
generally safe for concurrent reads when configured with connection
pooling adapters.

## 2. Classes

### `CachedResult`

Represents HTTP 304 Not Modified response with prior metadata.

Attributes:
path: File system path to the previously downloaded artifact.
sha256: SHA256 checksum associated with the cached artifact.
content_length: Size of the cached payload in bytes.
etag: Entity tag reported by the origin server, if any.
last_modified: Last-Modified header value supplied by the origin, if any.

Examples:
```python
>>> CachedResult(
...     path="/tmp/file.pdf",
...     sha256="abc",
...     content_length=1024,
...     etag="W/\"etag\"",
...     last_modified="Fri, 01 Jan 2021 00:00:00 GMT"
... )
CachedResult(path='/tmp/file.pdf', sha256='abc', content_length=1024, etag='W/\"etag\"', last_modified='Fri, 01 Jan 2021 00:00:00 GMT')
```

### `ModifiedResult`

Represents HTTP 200 response requiring fresh download.

Attributes:
etag: Entity tag reported by the origin server.
last_modified: Last-Modified header describing the remote resource timestamp.

Examples:
```python
>>> ModifiedResult(etag=None, last_modified=None)
ModifiedResult(etag=None, last_modified=None)
```

### `ConditionalRequestHelper`

Utility for constructing conditional requests and interpreting responses.

Attributes:
prior_etag: Previously observed entity tag for the resource.
prior_last_modified: Previously observed Last-Modified header value.
prior_sha256: Cached payload hash to validate 304 responses.
prior_content_length: Size of the cached payload in bytes.
prior_path: Local path where the cached artifact resides.

Examples:
```python
>>> helper = ConditionalRequestHelper(prior_etag="abcd", prior_path="/tmp/file.pdf")
>>> helper.build_headers()
{'If-None-Match': 'abcd'}
```

Methods:
- ``build_headers()``: Generate conditional request headers from cached metadata.
- ``interpret_response(response)``: Interpret HTTP responses as ``CachedResult`` or ``ModifiedResult`` and validate metadata for 304 outcomes.
