# 1. Module: conditional

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.conditional``.

Conditional HTTP request helpers for ETag and Last-Modified caching.

## 1. Functions

### `build_headers(self)`

Generate conditional request headers from prior metadata.

Args:
None

Returns:
Dictionary containing conditional HTTP headers suited for reuse in requests.

### `interpret_response(self, response)`

Interpret response status and headers as cached or modified result.

Args:
response: HTTP response returned from the conditional request.

Returns:
`CachedResult` when the origin reports HTTP 304, otherwise `ModifiedResult`.

Raises:
ValueError: If a 304 response arrives without complete prior metadata. The
exception lists missing fields to simplify debugging manifest issues.

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
>>> CachedResult(
...     path="/tmp/file.pdf",
...     sha256="abc",
...     content_length=1024,
...     etag="W/\"etag\"",
...     last_modified="Fri, 01 Jan 2021 00:00:00 GMT"
... )
CachedResult(path='/tmp/file.pdf', sha256='abc', content_length=1024, etag='W/"etag"', last_modified='Fri, 01 Jan 2021 00:00:00 GMT')

### `ModifiedResult`

Represents HTTP 200 response requiring fresh download.

Attributes:
etag: Entity tag reported by the origin server.
last_modified: Last-Modified header describing the remote resource timestamp.

Examples:
>>> ModifiedResult(etag=None, last_modified=None)
ModifiedResult(etag=None, last_modified=None)

### `ConditionalRequestHelper`

Utility for constructing conditional requests and interpreting responses.

Attributes:
prior_etag: Previously observed entity tag for the resource.
prior_last_modified: Previously observed Last-Modified header value.
prior_sha256: Cached payload hash to validate 304 responses.
prior_content_length: Size of the cached payload in bytes.
prior_path: Local path where the cached artifact resides.

Examples:
>>> helper = ConditionalRequestHelper(prior_etag="abcd", prior_path="/tmp/file.pdf")
>>> helper.build_headers()
{'If-None-Match': 'abcd'}
