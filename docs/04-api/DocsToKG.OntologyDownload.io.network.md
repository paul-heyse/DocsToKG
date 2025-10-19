# 1. Module: network

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.io.network``.

## 1. Overview

Networking utilities for ontology downloads.

## 2. Functions

### `_enforce_idn_safety(host)`

Validate internationalized hostnames and reject suspicious patterns.

### `_rebuild_netloc(parsed, ascii_host)`

Reconstruct URL netloc with a normalized hostname.

### `_cached_getaddrinfo(host)`

Resolve *host* using an expiring LRU cache to avoid repeated DNS lookups.

### `_prune_dns_cache(current_time)`

Expire stale DNS entries and enforce the cache size bound.

### `register_dns_stub(host, handler)`

Register a DNS stub callable for ``host`` used during testing.

### `clear_dns_stubs()`

Remove all registered DNS stubs and purge cached stub lookups.

### `validate_url_security(url, http_config)`

Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

### `retry_with_backoff(func)`

Execute ``func`` with exponential backoff until it succeeds.

### `log_memory_usage(logger)`

Emit debug-level memory usage snapshots when enabled.

### `_parse_retry_after(value)`

*No documentation available.*

### `_is_retryable_status(status_code)`

*No documentation available.*

### `is_retryable_error(exc)`

Return ``True`` when ``exc`` represents a retryable network failure.

### `request_with_redirect_audit()`

Issue an HTTP request while validating every redirect target.

### `_extract_correlation_id(logger)`

*No documentation available.*

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
expected_hash: Optional ``<algorithm>:<hex>`` string enforcing a known hash.
cancellation_token: Optional token for cooperative cancellation.
url_already_validated: When ``True``, assumes *url* has already passed
:func:`validate_url_security` checks and skips redundant
validations.

Returns:
DownloadResult describing the final artifact and metadata.

Raises:
PolicyError: If policy validation fails or limits are exceeded.
OntologyDownloadError: If retryable download mechanisms exhaust or IO fails.

### `_normalize(self, service, host)`

*No documentation available.*

### `lease(self)`

Yield a session associated with ``service``/``host`` and return it to the pool.

### `clear(self)`

Close and forget all pooled sessions (testing helper).

### `_reset_hashers(self)`

Initialise hashlib objects for all supported algorithms.

### `_seed_hashers_from_file(self, path)`

Update hashers with the bytes already present on disk.

### `_request_with_redirect_audit(self)`

Issue an HTTP request while validating every redirect target.

### `_preliminary_head_check(self, url, session)`

Probe the origin with HEAD to audit media type and size before downloading.

The HEAD probe allows the pipeline to abort before streaming large
payloads that exceed configured limits and to log early warnings for
mismatched Content-Type headers reported by the origin.

Args:
url: Fully qualified download URL resolved by the planner.
session: Prepared requests session used for outbound calls.
headers: Headers to include with the HEAD probe. When omitted the
downloader will send the polite header set merged with any
resolver-supplied headers.
token_consumed: Indicates whether the caller already consumed a
rate-limit token prior to invoking the HEAD request.
remaining_budget: Optional callable returning the remaining time
budget (in seconds) before the download timeout is reached.
timeout_callback: Optional callable invoked when the requested
backoff would exhaust the remaining timeout budget.

Returns:
Tuple ``(content_type, content_length)`` extracted from response
headers. Each element is ``None`` when the origin omits it.

Raises:
PolicyError: Propagates download policy errors encountered during the HEAD request.
DownloadFailure: Raised when the timeout budget is exhausted prior to completing the HEAD request.

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
PolicyError: If download policies are violated (e.g., invalid URLs or disallowed MIME types).
OntologyDownloadError: If filesystem errors occur.
requests.HTTPError: Propagated when HTTP status codes indicate failure.

Returns:
None

### `_resolved_content_metadata(current_downloader, manifest)`

*No documentation available.*

### `_verify_expected_checksum(digests)`

*No documentation available.*

### `_resolve_digests()`

*No documentation available.*

### `_clear_partial_files()`

*No documentation available.*

### `_raise_timeout(elapsed)`

*No documentation available.*

### `_remaining_budget()`

*No documentation available.*

### `_fail_for_timeout()`

*No documentation available.*

### `_enforce_timeout()`

*No documentation available.*

### `_stream_once()`

*No documentation available.*

### `_stream_once_with_timeout()`

*No documentation available.*

### `_retry_after_hint(exc)`

*No documentation available.*

### `_on_retry(attempt_number, exc, delay)`

*No documentation available.*

## 3. Classes

### `SessionPool`

Lightweight pool that reuses requests sessions per (service, host).

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
streamed_digests: Mapping of hash algorithm names to hex digests computed during streaming.

Examples:
>>> from pathlib import Path
>>> from DocsToKG.OntologyDownload.settings import DownloadConfiguration
>>> downloader = StreamingDownloader(
...     destination=Path("/tmp/ontology.owl"),
...     headers={},
...     http_config=DownloadConfiguration(),
...     previous_manifest={},
...     logger=logging.getLogger("test"),
... )
>>> downloader.status
'fresh'
