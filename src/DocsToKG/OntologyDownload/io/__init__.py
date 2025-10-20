"""Aggregated IO helpers for DocsToKG ontology downloads.

This subpackage bundles filesystem utilities (archive extraction, checksum
helpers), streaming download logic with resilient retry/back-off behaviour, and
rate limiting primitives shared across resolvers.  Re-exporting the most common
symbols from here keeps importing ergonomics simple for the rest of the
codebase.
"""

from ..errors import DownloadFailure
from ..net import configure_http_client, get_http_client, reset_http_client
from .filesystem import (
    extract_archive_safe,
    extract_tar_safe,
    extract_zip_safe,
    format_bytes,
    generate_correlation_id,
    mask_sensitive_data,
    sanitize_filename,
    sha256_file,
)
from .network import (
    RDF_MIME_ALIASES,
    RDF_MIME_FORMAT_LABELS,
    DownloadResult,
    download_stream,
    is_retryable_error,
    log_memory_usage,
    retry_with_backoff,
    validate_url_security,
)
from .rate_limit import (
    REGISTRY,
    RateLimiterRegistry,
    SharedTokenBucket,
    TokenBucket,
    apply_retry_after,
    get_bucket,
    reset,
)

__all__ = [
    "extract_archive_safe",
    "extract_tar_safe",
    "extract_zip_safe",
    "format_bytes",
    "generate_correlation_id",
    "mask_sensitive_data",
    "sanitize_filename",
    "sha256_file",
    "DownloadFailure",
    "DownloadResult",
    "configure_http_client",
    "get_http_client",
    "RDF_MIME_ALIASES",
    "RDF_MIME_FORMAT_LABELS",
    "download_stream",
    "is_retryable_error",
    "log_memory_usage",
    "retry_with_backoff",
    "validate_url_security",
    "RateLimiterRegistry",
    "SharedTokenBucket",
    "TokenBucket",
    "REGISTRY",
    "apply_retry_after",
    "get_bucket",
    "reset",
    "reset_http_client",
]
