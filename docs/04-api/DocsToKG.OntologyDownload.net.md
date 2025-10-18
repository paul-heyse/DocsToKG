# 1. Module: net

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.net``.

## 1. Overview

Networking utilities for DocsToKG ontology downloads.

## 2. Functions

### `retry_with_backoff(func)`

Execute ``func`` with exponential backoff until it succeeds.

### `log_memory_usage(logger)`

Emit debug-level memory usage snapshots when enabled.

### `_shared_bucket_path(http_config, key)`

Return the filesystem path for the shared token bucket state.

### `_is_retryable_status(status_code)`

*No documentation available.*

### `_get_bucket(host, http_config, service)`

Return a token bucket keyed by host and optional service name.

Args:
host: Hostname extracted from the download URL.
http_config: Download configuration providing base rate limits.
service: Logical service identifier enabling per-service overrides.

Returns:
TokenBucket instance shared across downloads for throttling, seeded
with either per-host defaults or service-specific overrides.

### `download_stream()`

Download ontology content with HEAD validation, rate limiting, caching, retries, and hash checks.

Args:
url: URL of the ontology document to download.
destination: Target file path for the downloaded content.
headers: HTTP headers forwarded to the download request.
previous_manifest: Manifest metadata from a prior run, used for caching.
http_config: Download configuration containing timeouts, limits, and rate controls.
cache_dir: Directory where intermediary cached files are stored.
logger: Logger adapter for structured download telemetry.
expected_media_type: Expected Content-Type for validation, if known.
service: Logical service identifier for per-service rate limiting.

Returns:
DownloadResult describing the final artifact and metadata.

Raises:
PolicyError: If policy validation fails or limits are exceeded.
OntologyDownloadError: If retryable download mechanisms exhaust or IO fails.

### `__getattr__(name)`

Lazily proxy pipeline helpers without incurring import cycles.

### `consume(self, tokens)`

Consume tokens from the bucket, sleeping until capacity is available.

Args:
tokens: Number of tokens required for the current download request.

Returns:
None

### `_acquire_file_lock(self, handle)`

*No documentation available.*

### `_release_file_lock(self, handle)`

*No documentation available.*

### `_deserialize_state(raw)`

*No documentation available.*

### `_write_state(handle, state)`

*No documentation available.*

### `_try_consume(self, tokens)`

*No documentation available.*

### `consume(self, tokens)`

Consume ``tokens`` from the shared bucket, waiting when insufficient.

### `_preliminary_head_check(self, url, session)`

Probe the origin with HEAD to audit media type and size before downloading.

The HEAD probe logs early warnings for mismatched Content-Type headers reported by the origin and returns metadata that downstream components can use to inform progress reporting.

Args:
url: Fully qualified download URL resolved by the planner.
session: Prepared requests session used for outbound calls.

Returns:
Tuple ``(content_type, content_length)`` extracted from response
headers. Each element is ``None`` when the origin omits it.

### `_validate_media_type(self, actual_content_type, expected_media_type, url)`

Validate that the received ``Content-Type`` header is acceptable, tolerating aliases.

RDF endpoints often return generic XML or Turtle aliases, so the
validator accepts a small set of known MIME variants while still
surfacing actionable warnings for unexpected types.

Args:
actual_content_type: Raw header value reported by the origin server.
expected_media_type: MIME type declared by resolver metadata.
url: Download URL logged when mismatches occur.

Returns:
None

### `__call__(self, url, output_file, pooch_logger)`

Stream ontology content to disk while enforcing download policies.

Args:
url: Secure download URL resolved by the planner.
output_file: Temporary filename managed by pooch during download.
pooch_logger: Logger instance supplied by pooch (unused).

Raises:
PolicyError: If download limits are exceeded.
OntologyDownloadError: If filesystem errors occur.
requests.HTTPError: Propagated when HTTP status codes indicate failure.

Returns:
None

### `_resolved_content_metadata()`

*No documentation available.*

## 3. Classes

### `DownloadResult`

Result metadata for a completed download operation.

Attributes:
path: Final file path where the ontology document was stored.
status: Download status (`fresh`, `updated`, or `cached`).
sha256: SHA-256 checksum of the downloaded artifact.
etag: HTTP ETag returned by the upstream server, when available.
last_modified: Upstream last-modified header value if provided.
content_type: Reported MIME type when available (HEAD or GET).
content_length: Reported content length when available.

Examples:
>>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None, None, None)
>>> result.status
'fresh'

### `TokenBucket`

Token bucket used to enforce per-host and per-service rate limits.

Each unique combination of host and logical service identifier receives
its own bucket so resolvers can honour provider-specific throttling
guidance without starving other endpoints.

Attributes:
rate: Token replenishment rate per second.
capacity: Maximum number of tokens the bucket may hold.
tokens: Current token balance available for consumption.
timestamp: Monotonic timestamp of the last refill.
lock: Threading lock protecting bucket state.

Examples:
>>> bucket = TokenBucket(rate_per_sec=2.0, capacity=4.0)
>>> bucket.consume(1.0)  # consumes immediately
>>> isinstance(bucket.tokens, float)
True

### `SharedTokenBucket`

Token bucket backed by a filesystem state file for multi-process usage.

### `StreamingDownloader`

Custom downloader supporting HEAD validation, conditional requests, resume, and caching.

The downloader shares a :mod:`requests` session so it can issue a HEAD probe
prior to streaming content, verifies Content-Type and Content-Length against
expectations, and persists ETag/Last-Modified headers for cache-friendly
revalidation.

Attributes:
destination: Final location where the ontology will be stored.
custom_headers: HTTP headers supplied by the resolver.
http_config: Download configuration governing retries and limits.
previous_manifest: Manifest from prior runs used for caching.
logger: Logger used for structured telemetry.
status: Final download status (`fresh`, `updated`, or `cached`).
response_etag: ETag returned by the upstream server, if present.
response_last_modified: Last-modified timestamp provided by the server.
expected_media_type: MIME type provided by the resolver for validation.

Examples:
>>> from pathlib import Path
>>> from DocsToKG.OntologyDownload import DownloadConfiguration
>>> downloader = StreamingDownloader(
...     destination=Path("/tmp/ontology.owl"),
...     headers={},
...     http_config=DownloadConfiguration(),
...     previous_manifest={},
...     logger=logging.getLogger("test"),
... )
>>> downloader.status
'fresh'
