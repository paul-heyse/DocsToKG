# 1. Module: conditional

This reference documents the DocsToKG module ``DocsToKG.ContentDownload.conditional``.

Conditional Request Metadata

This module centralises helper types for conditional HTTP requests that rely on
ETag and Last-Modified headers. It provides strongly typed dataclasses for
capturing cached responses alongside a small orchestration helper that builds
request headers and interprets origin responses. The utilities are used by the
content download pipeline to honour polite download practices while avoiding
unnecessary network transfers.

Key Features:
- Dataclasses that model cached and modified responses with checksum metadata.
- Helper for constructing `If-None-Match`/`If-Modified-Since` headers.
- Utilities for validating 304 responses against previously cached artefacts.

Dependencies:
- `requests`: Required for the `Response` type used when interpreting outcomes.

Usage:
    from DocsToKG.ContentDownload.conditional import ConditionalRequestHelper

    helper = ConditionalRequestHelper(prior_etag="abcd", prior_path="/tmp/file.pdf")
    headers = helper.build_headers()
    response = session.get(url, headers=headers)
    result = helper.interpret_response(response)

## 1. Functions

### `build_headers(self)`

Generate conditional request headers from cached metadata.

Args:
self: Helper instance containing cached HTTP metadata.

Returns:
Mapping of conditional header names to values ready for ``requests``.

Examples:
>>> helper = ConditionalRequestHelper(prior_etag="abcd", prior_last_modified="Fri, 01 Jan 2021 00:00:00 GMT")
>>> helper.build_headers() == {'If-None-Match': 'abcd', 'If-Modified-Since': 'Fri, 01 Jan 2021 00:00:00 GMT'}
True

### `interpret_response(self, response)`

Interpret response status and headers as cached or modified result.

Args:
response: HTTP response returned from the conditional request.

Returns:
`CachedResult` when the origin reports HTTP 304, otherwise `ModifiedResult`.

Raises:
ValueError: If a 304 response arrives without complete prior metadata.
TypeError: If ``response`` lacks the minimal ``status_code``/``headers`` API.

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
