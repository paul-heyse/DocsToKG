"""Aggregated IO helpers for DocsToKG ontology downloads.

This subpackage bundles filesystem utilities (archive extraction, checksum
helpers), streaming download logic with resilient retry/back-off behaviour, and
the pyrate-limiter backed throttling façade shared across resolvers.  Re-
exporting the most common symbols keeps importing ergonomics simple for the rest
of the codebase.
"""

from ..errors import DownloadFailure
from ..net import configure_http_client, get_http_client, reset_http_client
from .extraction_constraints import (
    CaseCollisionDetector,
    ExtractionGuardian,
    PreScanValidator,
    apply_default_permissions,
    normalize_path_unicode,
    validate_disk_space,
    validate_entry_compression_ratio,
    validate_entry_count,
    validate_entry_type,
    validate_path_constraints,
    validate_streaming_file_size,
)
from .extraction_integrity import (
    check_windows_portability,
    validate_archive_format,
)
from .extraction_policy import (
    ExtractionSettings,
    lenient_defaults,
    safe_defaults,
    strict_defaults,
)
from .extraction_telemetry import (
    ExtractionErrorCode,
    ExtractionMetrics,
    ExtractionTelemetryEvent,
    TelemetryKey,
    error_message,
)
from .filesystem import (
    extract_archive_safe,
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
from .probe import ProbeResult, probe_url
from .rate_limit import RateLimiterHandle, get_bucket, reset

__all__ = [
    "extract_archive_safe",
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
    "ProbeResult",
    "probe_url",
    "RateLimiterHandle",
    "get_bucket",
    "reset",
    "reset_http_client",
    # Core extraction policy & telemetry
    "ExtractionSettings",
    "safe_defaults",
    "lenient_defaults",
    "strict_defaults",
    "ExtractionErrorCode",
    "ExtractionTelemetryEvent",
    "ExtractionMetrics",
    "TelemetryKey",
    "error_message",
    # Core extraction constraints (Phase 2)
    "PreScanValidator",
    "CaseCollisionDetector",
    "normalize_path_unicode",
    "validate_entry_type",
    "validate_entry_count",
    "validate_entry_compression_ratio",
    "validate_file_size",
    "validate_path_constraints",
    "validate_streaming_file_size",
    # Phase 3-4 constraints
    "ExtractionGuardian",
    "validate_disk_space",
    "apply_default_permissions",
    # Core integrity checks
    "check_windows_portability",
    "validate_archive_format",
]
