# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_observability",
#   "purpose": "Observability, metrics, diagnostics, and error taxonomy for extraction",
#   "sections": [
#     {"id": "structured_metrics", "name": "Structured Metrics & Events", "anchor": "METRICS", "kind": "observability"},
#     {"id": "error_taxonomy", "name": "Failure Taxonomy", "anchor": "ERROR", "kind": "diagnostics"},
#     {"id": "libarchive_info", "name": "Libarchive Version Fingerprint", "anchor": "LIBARCH", "kind": "diagnostics"}
#   ]
# }
# === /NAVMAP ===

"""Observability, metrics, diagnostics for archive extraction.

Implements:
- Structured event logging with consistent field names
- Centralized error taxonomy and helpers
- Libarchive version fingerprinting
- Per-archive audit record with deterministic schema
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

import libarchive

from ..errors import ConfigError


# ============================================================================
# LIBARCHIVE VERSION FINGERPRINT
# ============================================================================


@dataclass
class LibarchiveInfo:
    """Libarchive version and build information."""

    version: str
    build_flags: Optional[str] = None
    platform: str = ""
    python_version: str = ""
    capture_time: float = field(default_factory=time.time)

    _cached_instance: Optional[LibarchiveInfo] = None

    @classmethod
    def get_singleton(cls) -> LibarchiveInfo:
        """Get cached libarchive info (computed once per process)."""
        if cls._cached_instance is None:
            try:
                version = libarchive.version
            except Exception:
                version = "unknown"

            cls._cached_instance = cls(
                version=version,
                platform=os.name,
            )

        return cls._cached_instance


# ============================================================================
# RUN CONTEXT
# ============================================================================


@dataclass
class ExtractionRunContext:
    """Single extraction run's context (captured once per archive)."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    archive_path: str = ""
    archive_sha256_short: str = ""  # First 12 chars
    encapsulated_root: str = ""
    format: str = ""
    filters: List[str] = field(default_factory=list)
    libarchive_version: str = ""
    mode: str = "extract"  # "extract" | "probe"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for log emissions."""
        return {
            "run_id": self.run_id,
            "archive": self.archive_path,
            "archive_sha256": self.archive_sha256_short,
            "encapsulated_root": self.encapsulated_root,
            "format": self.format,
            "filters": self.filters,
            "libarchive_version": self.libarchive_version,
            "mode": self.mode,
        }


# ============================================================================
# STRUCTURED METRICS
# ============================================================================


@dataclass
class PreScanMetrics:
    """Pre-scan phase metrics."""

    entries_total: int = 0
    entries_included: int = 0
    entries_skipped: int = 0
    bytes_declared: int = 0
    max_depth: int = 0
    size_mix: Dict[str, float] = field(
        default_factory=lambda: {"small_pct": 0, "medium_pct": 0, "large_pct": 0}
    )
    space_available_bytes: Optional[int] = None
    space_needed_bytes: Optional[int] = None
    space_margin: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExtractMetrics:
    """Extract phase metrics."""

    bytes_written: int = 0
    ratio_total: Optional[float] = None  # bytes_declared / archive_size
    prealloc_bytes_reserved: int = 0
    io_buffer_size_bytes: int = 0
    atomic_renames: int = 0
    atomic_dirfsyncs: int = 0
    hash_mode: str = "inline"
    hash_bytes_hashed: int = 0
    hash_algorithms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# ERROR TAXONOMY
# ============================================================================

# Comprehensive error code catalog
ERROR_CODES = {
    "E_TRAVERSAL": "Path traversal attempt detected",
    "E_LINK_TYPE": "Symlink or hardlink not permitted",
    "E_BOMB_RATIO": "Compression ratio exceeds limit",
    "E_PORTABILITY": "Windows reserved name or invalid path",
    "E_UNSUPPORTED_FORMAT": "Archive format not supported",
    "E_SPACE": "Insufficient disk space",
    "E_TIMEOUT": "Extraction wall-time exceeded",
    "E_PREALLOC": "File preallocation failed",
    "E_WRITE": "Write to file failed",
    "E_FSYNC_FILE": "File fsync failed",
    "E_FSYNC_DIR": "Directory fsync failed",
    "E_ENTRY_RATIO": "Per-entry compression ratio exceeded",
    "E_FILE_SIZE": "File size exceeds limit",
    "E_FILE_SIZE_STREAM": "File size exceeded during streaming",
    "E_DUP_ENTRY": "Duplicate entry detected",
    "E_CASEFOLD_COLLISION": "Case-insensitive collision detected",
    "E_DEPTH": "Path depth exceeds limit",
    "E_SEGMENT_LEN": "Path component too long",
    "E_PATH_LEN": "Total path length exceeds limit",
    "E_UNICODE_DECODE": "Unicode decoding failed",
    "E_CRC_MISMATCH": "CRC or digest mismatch",
    "E_SIZE_MISMATCH": "Declared size doesn't match written",
    "E_SHORT_READ": "Premature EOF from archive",
    "E_SPECIAL_TYPE": "Device, FIFO, or socket not permitted",
    "E_MULTI_TOP": "Multiple top-level directories in archive",
    "E_POLICY_INVALID": "Extraction policy is invalid",
    "E_OBSERVABILITY": "Failed to emit observability event",
}


@dataclass
class ExtractionError:
    """Standardized extraction error."""

    error_code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    entry_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "entry_name": self.entry_name,
        }


class ExtractionErrorHelper:
    """Centralized error emission and logging."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error helper.

        Args:
            logger: Logger instance for structured logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.last_error: Optional[ExtractionError] = None

    def emit_error(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        entry_name: Optional[str] = None,
        context: Optional[ExtractionRunContext] = None,
        duration_ms: Optional[float] = None,
    ) -> ExtractionError:
        """Emit a standardized error.

        Args:
            error_code: Error code from ERROR_CODES
            message: Human-readable message
            details: Additional details dictionary
            entry_name: Entry that caused the error (if applicable)
            context: Run context for common fields
            duration_ms: Extraction duration so far

        Returns:
            ExtractionError instance
        """
        error = ExtractionError(
            error_code=error_code,
            message=message,
            details=details or {},
            entry_name=entry_name,
        )

        self.last_error = error

        # Build structured log
        log_data = {
            "type": "extract.error",
            "error_code": error_code,
            "message": message,
            "details": error.details,
        }

        if context:
            log_data.update(context.to_dict())
            log_data["duration_ms"] = duration_ms or 0

        if entry_name:
            log_data["entry_name"] = entry_name

        self.logger.error("Extraction error", extra=log_data)

        return error


# ============================================================================
# EVENT EMISSION
# ============================================================================


class ExtractionEventEmitter:
    """Emits structured extraction events."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize event emitter.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = time.time()

    def _build_common_fields(
        self,
        context: ExtractionRunContext,
    ) -> Dict[str, Any]:
        """Build common fields for all events."""
        duration_ms = (time.time() - self.start_time) * 1000
        return {
            **context.to_dict(),
            "duration_ms": duration_ms,
        }

    def emit_start(self, context: ExtractionRunContext) -> None:
        """Emit extraction start event."""
        log_data = {
            "type": "extract.start",
            **self._build_common_fields(context),
        }
        self.logger.info("Extraction started", extra=log_data)

    def emit_pre_scan_done(
        self,
        context: ExtractionRunContext,
        metrics: PreScanMetrics,
    ) -> None:
        """Emit pre-scan completion event."""
        log_data = {
            "type": "extract.pre_scan.done",
            **self._build_common_fields(context),
            **metrics.to_dict(),
        }
        self.logger.info("Pre-scan completed", extra=log_data)

    def emit_extract_start(self, context: ExtractionRunContext) -> None:
        """Emit extraction phase start event."""
        log_data = {
            "type": "extract.extract.start",
            **self._build_common_fields(context),
        }
        self.logger.info("Extraction phase started", extra=log_data)

    def emit_extract_done(
        self,
        context: ExtractionRunContext,
        metrics: ExtractMetrics,
    ) -> None:
        """Emit extraction completion event."""
        log_data = {
            "type": "extract.extract.done",
            **self._build_common_fields(context),
            **metrics.to_dict(),
        }
        self.logger.info("Extraction completed", extra=log_data)

    def emit_audit(
        self,
        context: ExtractionRunContext,
        audit_path: str,
        entries_count: int,
        bytes_written: int,
    ) -> None:
        """Emit audit manifest emission event."""
        log_data = {
            "type": "extract.audit.emitted",
            **self._build_common_fields(context),
            "path": audit_path,
            "entries": entries_count,
            "bytes": bytes_written,
        }
        self.logger.info("Audit manifest emitted", extra=log_data)

    def emit_libarchive_info(self) -> None:
        """Emit libarchive version info (one-time per process)."""
        libarch_info = LibarchiveInfo.get_singleton()
        log_data = {
            "type": "extract.libarchive.info",
            "version": libarch_info.version,
            "platform": libarch_info.platform,
            "pid": os.getpid(),
        }
        self.logger.info("Libarchive info", extra=log_data)
