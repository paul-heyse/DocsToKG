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
    validate_file_size,
    validate_path_constraints,
    validate_streaming_file_size,
)
from .extraction_policy import (
    ExtractionPolicy,
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
from .rate_limit import RateLimiterHandle, apply_retry_after, get_bucket, reset
from .extraction_throughput import (
    CPUGuard,
    HashingPipeline,
    atomic_rename_and_fsync,
    compute_adaptive_buffer_size,
    create_temp_path,
    preallocate_file,
    should_extract_entry,
)
from .extraction_integrity import (
    DuplicateDetector,
    DuplicateEntry,
    IntegrityCheckResult,
    IntegrityVerifier,
    ManifestEntry,
    ProvenanceManifest,
    TimestampPolicy,
    apply_mtime,
    compute_target_mtime,
    get_sort_key,
    normalize_pathname,
    validate_format_allowed,
)
from .extraction_observability import (
    ExtractionErrorHelper,
    ExtractionEventEmitter,
    ExtractionRunContext,
    ExtractMetrics,
    ExtractionError,
    ERROR_CODES,
    LibarchiveInfo,
    PreScanMetrics,
)
from .extraction_extensibility import (
    ArchiveProbe,
    EntryMeta,
    IdempotenceHandler,
    IdempotenceStats,
    PortabilityChecker,
    PolicyBuilder,
    WINDOWS_RESERVED_NAMES,
)

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
    "RateLimiterHandle",
    "apply_retry_after",
    "get_bucket",
    "reset",
    "reset_http_client",
    # New extraction policy & telemetry
    "ExtractionPolicy",
    "safe_defaults",
    "lenient_defaults",
    "strict_defaults",
    "ExtractionErrorCode",
    "ExtractionTelemetryEvent",
    "ExtractionMetrics",
    "TelemetryKey",
    "error_message",
    # New extraction constraints (Phase 2)
    "PreScanValidator",
    "CaseCollisionDetector",
    "normalize_path_unicode",
    "validate_entry_type",
    "validate_entry_count",
    "validate_entry_compression_ratio",
    "validate_file_size",
    "validate_path_constraints",
    "validate_streaming_file_size",
    # New Phase 3-4 constraints
    "ExtractionGuardian",
    "validate_disk_space",
    "apply_default_permissions",
    # New throughput optimizations (Phase 3-4)
    "compute_adaptive_buffer_size",
    "preallocate_file",
    "create_temp_path",
    "atomic_rename_and_fsync",
    "HashingPipeline",
    "should_extract_entry",
    "CPUGuard",
    # New correctness & integrity (Phase 3-4)
    "IntegrityVerifier",
    "IntegrityCheckResult",
    "TimestampPolicy",
    "compute_target_mtime",
    "apply_mtime",
    "normalize_pathname",
    "validate_format_allowed",
    "get_sort_key",
    "DuplicateDetector",
    "DuplicateEntry",
    "ProvenanceManifest",
    "ManifestEntry",
    # New observability (Phase 4)
    "ExtractionErrorHelper",
    "ExtractionEventEmitter",
    "ExtractionRunContext",
    "ExtractMetrics",
    "PreScanMetrics",
    "ExtractionError",
    "ERROR_CODES",
    "LibarchiveInfo",
    # New extensibility (Phase 4)
    "ArchiveProbe",
    "EntryMeta",
    "IdempotenceHandler",
    "IdempotenceStats",
    "PortabilityChecker",
    "PolicyBuilder",
    "WINDOWS_RESERVED_NAMES",
]
