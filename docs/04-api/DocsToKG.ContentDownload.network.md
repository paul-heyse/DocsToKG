# 1. Module: network

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.networking``.

## 1. Overview

Unified Network Utilities

This module consolidates HTTP retry helpers, conditional request utilities, and
session construction logic into a single import surface. Historically these
helpers lived in separate ``http`` and ``conditional`` modules; co-locating
them reduces cross-module bootstrapping and ensures shared defaults remain
aligned.

Key Features:

- ``create_session``: Configure ``requests.Session`` instances with pooled
  adapters and optional header injection.
- ``request_with_retries``: Execute resilient HTTP calls with jittered,
  exponential backoff while honouring ``Retry-After`` directives.
- ``ConditionalRequestHelper``: Build and validate conditional request headers
  for polite revalidation workflows.

Usage:

    from DocsToKG.ContentDownload.networking import (
        ConditionalRequestHelper,
        create_session,
        request_with_retries,
    )

    session = create_session({"User-Agent": "DocsToKG/1.0"})
    response = request_with_retries(session, "GET", "https://example.org/resource")
    helper = ConditionalRequestHelper(prior_etag="abc123")
    headers = helper.build_headers()

Args:
    None.

Returns:
    None.

Raises:
    None.

## 2. Functions

### `create_session(headers)`

Return a ``requests.Session`` configured for DocsToKG network requests.

Args:
headers: Optional mapping of HTTP headers applied to the session. The mapping
is copied into the session's header store so caller-owned dictionaries remain
unchanged.
adapter_max_retries: Retry count configured on mounted HTTP adapters. Defaults to
``0`` so :func:`request_with_retries` governs retry behaviour.
pool_connections: Lower bound of connection pools shared across the session's adapters.
pool_maxsize: Maximum number of connections kept per host in the adapter pool.

Returns:
requests.Session: Session instance with HTTP/HTTPS adapters mounted and ready for pipeline usage.

Raises:
None.

Notes:
The returned session uses ``HTTPAdapter`` for connection pooling. It is safe to share
across threads provided callers avoid mutating shared mutable state (for example,
updating ``session.headers``) once the session is distributed to worker threads.

### `parse_retry_after_header(response)`

Parse ``Retry-After`` header and return wait time in seconds.

Args:
response (requests.Response): HTTP response potentially containing a
``Retry-After`` header.

Returns:
float | None: Seconds the caller should wait before retrying, or
``None`` when the header is absent or invalid.

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
session: Session used to execute the outbound request.
method: HTTP method such as ``"GET"`` or ``"HEAD"``.
url: Fully qualified URL for the request.
max_retries: Maximum number of retry attempts before returning the final response or
raising an exception. Defaults to ``3``.
retry_statuses: HTTP status codes that should trigger a retry. Defaults to
``{429, 500, 502, 503, 504}``.
backoff_factor: Base multiplier for exponential backoff delays in seconds. Defaults to ``0.75``.
respect_retry_after: Whether to parse and obey ``Retry-After`` headers. Defaults to ``True``.
**kwargs: Additional keyword arguments forwarded directly to :meth:`requests.Session.request`.

Returns:
requests.Response: Successful response object. Callers are responsible for closing the
response when streaming content.

Raises:
ValueError: If ``max_retries`` or ``backoff_factor`` are invalid or ``url``/``method`` are empty.
requests.RequestException: If all retry attempts fail due to network errors or the session raises an exception.

### `head_precheck(session, url, timeout)`

Issue a lightweight request to determine whether ``url`` returns a PDF.

The helper primarily relies on a HEAD request capped to a short timeout.
Some providers respond with ``405`` or ``501`` for HEAD requests; in those
cases a secondary streaming GET probe is issued that reads at most one
chunk to infer the payload type without downloading the entire resource.

Args:
session: HTTP session used for outbound requests.
url: Candidate download URL.
timeout: Maximum time budget in seconds for the probe.

Returns:
``True`` when the response appears to represent a binary payload such as
a PDF. ``False`` when the response clearly corresponds to HTML or an
error status. Any network exception results in ``True`` to avoid
blocking legitimate downloads.

### `_looks_like_pdf(headers)`

Return ``True`` when response headers suggest a PDF payload.

### `_head_precheck_via_get(session, url, timeout)`

Fallback GET probe for providers that reject HEAD requests.

### `request_func()`

Invoke :meth:`requests.Session.request` on the provided session.

### `build_headers(self)`

Generate conditional request headers from cached metadata.

Args:
None

Returns:
Mapping[str, str]: Headers suitable for ``requests`` invocations.

### `interpret_response(self, response)`

Classify origin responses as cached or modified results.

Args:
response (requests.Response): HTTP response returned from the
conditional request.

Returns:
CachedResult | ModifiedResult: Cached metadata when the origin
reports HTTP 304, otherwise wrapped metadata from a fresh download.

Raises:
ValueError: If a 304 response arrives without complete cached
metadata.
TypeError: If ``response`` lacks ``status_code`` or ``headers``.

## 3. Classes

### `CachedResult`

Represents an HTTP 304 response resolved via cached metadata.

Attributes:
path (str): Filesystem path that stores the cached artifact.
sha256 (str): SHA-256 checksum associated with the cached payload.
content_length (int): Size of the cached payload in bytes.
etag (str | None): Entity tag reported by the origin server.
last_modified (str | None): ``Last-Modified`` timestamp provided by the
origin server.

Examples:
>>> CachedResult(path="/tmp/file.pdf", sha256="abc", content_length=10, etag=None, last_modified=None)
CachedResult(path='/tmp/file.pdf', sha256='abc', content_length=10, etag=None, last_modified=None)

### `ModifiedResult`

Represents a fresh HTTP 200 response requiring download.

Attributes:
etag (str | None): Entity tag reported by the origin server.
last_modified (str | None): ``Last-Modified`` timestamp describing the
remote resource.

Examples:
>>> ModifiedResult(etag="abc", last_modified="Tue, 15 Nov 1994 12:45:26 GMT")
ModifiedResult(etag='abc', last_modified='Tue, 15 Nov 1994 12:45:26 GMT')

### `ConditionalRequestHelper`

Construct headers and interpret responses for conditional requests.

The helper encapsulates cached metadata (ETag, Last-Modified, hashes) so the
caller can generate polite conditional headers and validate upstream 304
responses before reusing cached artefacts.

Attributes:
prior_etag: Cached entity tag from a previous download.
prior_last_modified: Cached ``Last-Modified`` header value.
prior_sha256: SHA-256 checksum of the cached content.
prior_content_length: Cached payload length in bytes.
prior_path: Filesystem path storing the cached artefact.

Examples:
>>> helper = ConditionalRequestHelper(prior_etag="abc123")
>>> helper.build_headers()
{'If-None-Match': 'abc123'}
