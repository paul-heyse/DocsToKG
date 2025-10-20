# 1. Module: io

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.io``.

## 1. Overview

Filesystem safety, rate limiting, and networking helpers for ontology downloads.

## 2. Functions

### `_resolve_max_uncompressed_bytes(limit)`

Return the effective archive expansion limit, honoring runtime overrides.

### `sanitize_filename(filename)`

Return a filesystem-safe filename derived from ``filename``.

### `generate_correlation_id()`

Return a short-lived identifier that links related log entries.

### `mask_sensitive_data(payload)`

Return a copy of ``payload`` with common secret fields masked.

### `_enforce_idn_safety(host)`

Validate internationalized hostnames and reject suspicious patterns.

### `_rebuild_netloc(parsed, ascii_host)`

Reconstruct URL netloc with a normalized hostname.

### `_cached_getaddrinfo(host)`

Resolve *host* using an expiring LRU cache to avoid repeated DNS lookups.

### `_prune_dns_cache(current_time)`

Expire stale DNS entries and enforce the cache size bound.

### `validate_url_security(url, http_config)`

Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

### `sha256_file(path)`

Compute the SHA-256 digest for the provided file.

### `_compute_file_hash(path, algorithm)`

Compute ``algorithm`` digest for ``path``.

### `_validate_member_path(member_name)`

Validate archive member paths to prevent traversal attacks.

### `_check_compression_ratio()`

Ensure compressed archives do not expand beyond the permitted ratio.

### `_enforce_uncompressed_ceiling()`

Ensure uncompressed payload stays within configured limits.

### `extract_zip_safe(zip_path, destination)`

Extract a ZIP archive while preventing traversal and compression bombs.

### `extract_tar_safe(tar_path, destination)`

Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

### `extract_archive_safe(archive_path, destination)`

Extract archives by dispatching to the appropriate safe handler.

### `_shared_bucket_path(http_config, key)`

Return the filesystem path for the shared token bucket state.

### `get_bucket()`

Return a registry-managed bucket.

### `apply_retry_after()`

Adjust bucket capacity after receiving a Retry-After hint.

### `reset()`

Clear all buckets (testing hook).

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

### `_materialize_cached_file(source, destination)`

Link or move ``source`` into ``destination`` without redundant copies.

Returns a tuple ``(artifact_path, cache_path)`` where ``artifact_path`` is the final
destination path and ``cache_path`` points to the retained cache file (which may be
identical to ``artifact_path`` when the cache entry is moved instead of linked).

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

Returns:
DownloadResult describing the final artifact and metadata.

Raises:
PolicyError: If policy validation fails or limits are exceeded.
OntologyDownloadError: If retryable download mechanisms exhaust or IO fails.

### `__getattr__(name)`

Lazily proxy pipeline helpers without incurring import cycles.

### `format_bytes(num)`

Return a human-readable representation for ``num`` bytes.

### `_mask_value(value, key_hint)`

*No documentation available.*

### `consume(self, tokens)`

Consume tokens from the bucket, sleeping until capacity is available.

### `_acquire_file_lock(self, handle)`

*No documentation available.*

### `_release_file_lock(self, handle)`

*No documentation available.*

### `_read_state(self, handle)`

*No documentation available.*

### `_write_state(self, handle, state)`

*No documentation available.*

### `_try_consume(self, tokens)`

*No documentation available.*

### `consume(self, tokens)`

Consume tokens from the shared bucket, waiting when insufficient.

### `_normalize(self, service, host)`

*No documentation available.*

### `lease(self)`

Yield a session associated with ``service``/``host`` and return it to the pool.

### `clear(self)`

Close and forget all pooled sessions (testing helper).

### `_qualify(self, service, host)`

*No documentation available.*

### `_normalize_rate(self)`

*No documentation available.*

### `get_bucket(self)`

Return a token bucket for ``service``/``host`` using shared registry.

### `apply_retry_after(self)`

Reduce available tokens to honor server-provided retry-after hints.

### `reset(self)`

Clear all registered buckets (used in tests).

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

### `_resolved_content_metadata(current_downloader, manifest)`

*No documentation available.*

### `_verify_expected_checksum(sha256_value)`

*No documentation available.*

### `_stream_once()`

*No documentation available.*

### `_retry_after_hint(exc)`

*No documentation available.*

### `_on_retry(attempt_number, exc, delay)`

*No documentation available.*

### `_clear_partial_files()`

*No documentation available.*

## 3. Classes

### `TokenBucket`

Token bucket used to enforce per-host and per-service rate limits.

### `SharedTokenBucket`

Token bucket backed by a filesystem state file for multi-process usage.

### `Shared HTTP client`

OntologyDownload reuses a singleton :mod:`httpx` client supplied by
``DocsToKG.OntologyDownload.net``. The client is decorated with a Hishel cache
under ``CACHE_DIR/http/ontology`` and participates in polite header injection,
redirect auditing, and Retry-After aware throttling. Use
``configure_http_client`` to inject a custom ``httpx.Client`` (for tests or
advanced transports) and ``reset_http_client`` to restore defaults.

### `_BucketEntry`

*No documentation available.*

### `RateLimiterRegistry`

Manage shared token buckets keyed by (service, host).

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

The downloader reuses the shared :mod:`httpx` client so it can issue a HEAD
probe prior to streaming content, verifies ``Content-Type`` and ``Content-Length``
against expectations, and persists ETag/Last-Modified headers for cache-friendly
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
