# 1. Module: download

This reference documents the DocsToKG module ``DocsToKG.OntologyDownload.download``.

Ontology download utilities.

This module exposes the hardened download primitives used by the refactored
pipeline. Features include per-host and per-service rate limiting, polite
header construction, manifest-aware caching, resilient streaming downloads with
HTTP fallback classification, and centralized archive extraction with traversal
and compression-bomb protection. These helpers back the automatic resolver
fallback and streaming normalization workflows defined in the openspec.

## 1. Functions

### `_log_memory(logger, event)`

Emit debug-level memory usage snapshots when enabled.

Args:
logger: Logger instance controlling verbosity for download telemetry.
event: Short label describing the lifecycle point (e.g., ``before``).

Returns:
None

### `_is_retryable_status(status_code)`

*No documentation available.*

### `sanitize_filename(filename)`

Sanitize filenames to prevent directory traversal and unsafe characters.

Args:
filename: Candidate filename provided by an upstream service.

Returns:
Safe filename compatible with local filesystem storage.

### `_enforce_idn_safety(host)`

Validate internationalized hostnames and reject suspicious patterns.

Args:
host: Hostname component extracted from the download URL.

Returns:
None

Raises:
ConfigError: If the hostname mixes multiple scripts or contains invisible characters.

### `_rebuild_netloc(parsed, ascii_host)`

Reconstruct URL netloc with a normalized hostname.

Args:
parsed: Parsed URL components produced by :func:`urllib.parse.urlparse`.
ascii_host: ASCII-normalized hostname (potentially IPv6).

Returns:
String suitable for use as the netloc portion of a URL.

### `validate_url_security(url, http_config)`

Validate URLs to avoid SSRF, enforce HTTPS, normalize IDNs, and honor host allowlists.

Hostnames are converted to punycode before resolution, and both direct IP
addresses and DNS results are rejected when they target private or loopback
ranges to prevent server-side request forgery.

Args:
url: URL returned by a resolver for ontology download.
http_config: Download configuration providing optional host allowlist.

Returns:
HTTPS URL safe for downstream download operations.

Raises:
ConfigError: If the URL violates security requirements or allowlists.

### `sha256_file(path)`

Compute the SHA-256 digest for the provided file.

Args:
path: Path to the file whose digest should be calculated.

Returns:
Hexadecimal SHA-256 checksum string.

### `_validate_member_path(member_name)`

Validate archive member paths to prevent traversal attacks.

Args:
member_name: Path declared within the archive.

Returns:
Sanitised relative path safe for extraction on the local filesystem.

Raises:
ConfigError: If the member path is absolute or contains traversal segments.

### `_check_compression_ratio()`

Ensure compressed archives do not expand beyond the permitted ratio.

Args:
total_uncompressed: Sum of file sizes within the archive.
compressed_size: Archive file size on disk (or sum of compressed entries).
archive: Path to the archive on disk.
logger: Optional logger for emitting diagnostic messages.
archive_type: Human readable label for error messages (ZIP/TAR).

Raises:
ConfigError: If the archive exceeds the allowed expansion ratio.

### `extract_zip_safe(zip_path, destination)`

Extract a ZIP archive while preventing traversal and compression bombs.

Args:
zip_path: Path to the ZIP file to extract.
destination: Directory where extracted files should be stored.
logger: Optional logger for emitting extraction telemetry.

Returns:
List of extracted file paths.

Raises:
ConfigError: If the archive contains unsafe paths, compression bombs, or is missing.

### `extract_tar_safe(tar_path, destination)`

Safely extract tar archives (tar, tar.gz, tar.xz) with traversal and compression checks.

Args:
tar_path: Path to the tar archive (tar, tar.gz, tar.xz).
destination: Directory where extracted files should be stored.
logger: Optional logger for emitting extraction telemetry.

Returns:
List of extracted file paths.

Raises:
ConfigError: If the archive is missing, unsafe, or exceeds compression limits.

### `extract_archive_safe(archive_path, destination)`

Extract archives by dispatching to the appropriate safe handler.

Args:
archive_path: Path to the archive on disk.
destination: Directory where files should be extracted.
logger: Optional logger receiving structured extraction telemetry.

Returns:
List of paths extracted from the archive in the order processed.

Raises:
ConfigError: If the archive format is unsupported or extraction fails.

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
ConfigError: If validation fails, limits are exceeded, or HTTP errors occur.

### `consume(self, tokens)`

Consume tokens from the bucket, sleeping until capacity is available.

Args:
tokens: Number of tokens required for the current download request.

Returns:
None

### `_preliminary_head_check(self, url, session)`

Probe the origin with HEAD to audit media type and size before downloading.

The HEAD probe allows the pipeline to abort before streaming large
payloads that exceed configured limits and to log early warnings for
mismatched Content-Type headers reported by the origin.

Args:
url: Fully qualified download URL resolved by the planner.
session: Prepared requests session used for outbound calls.

Returns:
Tuple ``(content_type, content_length)`` extracted from response
headers. Each element is ``None`` when the origin omits it.

Raises:
ConfigError: If the origin reports a payload larger than the
configured ``max_download_size_gb`` limit.

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
ConfigError: If download limits are exceeded or filesystem errors occur.
requests.HTTPError: Propagated when HTTP status codes indicate failure.

Returns:
None

## 2. Classes

### `DownloadResult`

Result metadata for a completed download operation.

Attributes:
path: Final file path where the ontology document was stored.
status: Download status (`fresh`, `updated`, or `cached`).
sha256: SHA-256 checksum of the downloaded artifact.
etag: HTTP ETag returned by the upstream server, when available.
last_modified: Upstream last-modified header value if provided.

Examples:
>>> result = DownloadResult(Path("ontology.owl"), "fresh", "deadbeef", None, None)
>>> result.status
'fresh'

### `DownloadFailure`

Raised when an HTTP download attempt fails.

Attributes:
status_code: Optional HTTP status code returned by the upstream service.
retryable: Whether the failure is safe to retry with an alternate resolver.

Examples:
>>> raise DownloadFailure("Unavailable", status_code=503, retryable=True)
Traceback (most recent call last):
DownloadFailure: Unavailable

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
>>> from DocsToKG.OntologyDownload.config import DownloadConfiguration
>>> downloader = StreamingDownloader(
...     destination=Path("/tmp/ontology.owl"),
...     headers={},
...     http_config=DownloadConfiguration(),
...     previous_manifest={},
...     logger=logging.getLogger("test"),
... )
>>> downloader.status
'fresh'
