# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_telemetry",
#   "purpose": "Error codes, telemetry constants, and structured logging for archive extraction policies",
#   "sections": [
#     {"id": "errors", "name": "Error Codes", "anchor": "ERR", "kind": "constants"},
#     {"id": "telemetry", "name": "Telemetry Keys", "anchor": "TEL", "kind": "constants"},
#     {"id": "logging", "name": "Structured Logging", "anchor": "LOG", "kind": "helpers"}
#   ]
# }
# === /NAVMAP ===

"""Error codes, telemetry constants, and structured logging for archive extraction policies.

This module defines the complete error taxonomy for the 10 hardening policies and provides
structured logging utilities to capture telemetry for both success and failure paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# ============================================================================
# ERROR CODES (15 total, mapped to policies)
# ============================================================================


class ExtractionErrorCode(str, Enum):
    """Error codes for archive extraction policy violations.

    Maps directly to specific policies and rejection reasons.
    """

    # Encapsulation & Root Management
    OVERWRITE_ROOT = "E_OVERWRITE_ROOT"  # Root collision (policy: reject)

    # DirFD & Race-Free Operations
    TRAVERSAL = "E_TRAVERSAL"  # Path escapes root
    OVERWRITE_FILE = "E_OVERWRITE_FILE"  # File exists (policy: reject)

    # Symlink & Hardlink Defense
    LINK_TYPE = "E_LINK_TYPE"  # Symlink or hardlink entry

    # Device/FIFO/Socket Quarantine
    SPECIAL_TYPE = "E_SPECIAL_TYPE"  # Device/FIFO/socket

    # Case-Fold Collision
    CASEFOLD_COLLISION = "E_CASEFOLD_COLLISION"  # Duplicate after casefold

    # Path Constraints
    DEPTH = "E_DEPTH"  # Path exceeds max depth
    SEGMENT_LEN = "E_SEGMENT_LEN"  # Component too long
    PATH_LEN = "E_PATH_LEN"  # Full path too long

    # Entry Budget
    ENTRY_BUDGET = "E_ENTRY_BUDGET"  # Too many entries

    # Per-File Size Guard
    FILE_SIZE = "E_FILE_SIZE"  # Declared size exceeds limit
    FILE_SIZE_STREAM = "E_FILE_SIZE_STREAM"  # Streamed size exceeds limit

    # Per-Entry Compression Ratio
    ENTRY_RATIO = "E_ENTRY_RATIO"  # Per-entry compression ratio exceeded

    # Space Budgeting
    SPACE = "E_SPACE"  # Insufficient disk space
    
    # Format & Filter Validation
    FORMAT_NOT_ALLOWED = "E_FORMAT_NOT_ALLOWED"  # Archive format/filter not in allow-list
    
    # Windows Portability
    PORTABILITY = "E_PORTABILITY"  # Windows reserved name or portability violation
    
    # Archive Corruption
    EXTRACT_CORRUPT = "E_EXTRACT_CORRUPT"  # Archive is corrupted or truncated
    EXTRACT_IO = "E_EXTRACT_IO"  # I/O error during extraction
    BOMB_RATIO = "E_BOMB_RATIO"  # Global compression ratio exceeded (zip bomb)


# ============================================================================
# TELEMETRY KEYS (Standard across all extraction operations)
# ============================================================================


class TelemetryKey(str, Enum):
    """Standard telemetry keys for extraction operations."""

    # Core context
    STAGE = "stage"  # Always "extract"
    ARCHIVE = "archive"  # Archive path
    FORMAT = "format"  # Auto-detected format (zip, tar, etc.)
    FILTERS = "filters"  # Compression filters applied

    # Entry & Size Metrics
    ENTRIES_TOTAL = "entries_total"  # Total entries in archive
    ENTRIES_ALLOWED = "entries_allowed"  # Entries passing pre-scan
    BYTES_DECLARED = "bytes_declared"  # Total uncompressed size
    BYTES_WRITTEN = "bytes_written"  # Actual bytes written
    RATIO_TOTAL = "ratio_total"  # Total compression ratio

    # Encapsulation & DirFD
    ENCAPSULATED_ROOT = "encapsulated_root"  # Root subdirectory path
    ENCAPSULATION_POLICY = "encapsulation_policy"  # sha256 | basename
    DIRFD = "dirfd"  # Whether DirFD operations used
    FILE_OPEN_FLAGS = "file_open_flags"  # O_NOFOLLOW, O_EXCL, etc.

    # Policies Applied
    POLICIES_APPLIED = "policies_applied"  # List of enabled policies

    # Error & Failure Info
    ERROR_CODE = "error_code"  # Error code (E_*)
    ERROR_REASON = "error_reason"  # Human-readable reason
    PARTIAL = "partial"  # true if partial extraction occurred

    # Performance
    DURATION_MS = "duration_ms"  # Extraction duration (milliseconds)


# ============================================================================
# TELEMETRY EVENT STRUCTURES
# ============================================================================


@dataclass
class ExtractionTelemetryEvent:
    """Structured telemetry event for an extraction operation.

    Captures all relevant context for success and failure cases.
    """

    stage: str = "extract"
    archive: str = ""
    format: str = ""
    filters: list[str] = field(default_factory=list)

    entries_total: int = 0
    entries_allowed: int = 0
    bytes_declared: int = 0
    bytes_written: int = 0
    ratio_total: float = 0.0

    encapsulated_root: str = ""
    encapsulation_policy: str = "sha256"
    dirfd: bool = False
    file_open_flags: list[str] = field(default_factory=list)

    policies_applied: list[str] = field(default_factory=list)

    error_code: Optional[str] = None
    error_reason: Optional[str] = None
    partial: bool = False

    duration_ms: float = 0.0
    
    # Provenance & reproducibility
    run_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    config_hash: str = ""
    format_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None and v != [] and v != 0.0}


@dataclass
class ExtractionMetrics:
    """Aggregated metrics for extraction operations."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_entries: int = 0
    entries_allowed: int = 0
    entries_extracted: int = 0
    entries_rejected: int = 0
    total_bytes: int = 0
    rejection_reason: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def finalize(self) -> None:
        """Mark metrics as complete."""
        self.end_time = time.time()


# ============================================================================
# ERROR MESSAGE HELPERS
# ============================================================================


def error_message(code: ExtractionErrorCode, detail: str = "") -> str:
    """Generate a descriptive error message for an error code.

    Args:
        code: The error code
        detail: Additional detail to append

    Returns:
        Human-readable error message
    """
    messages = {
        ExtractionErrorCode.OVERWRITE_ROOT: "Encapsulation root already exists",
        ExtractionErrorCode.TRAVERSAL: "Path traversal detected",
        ExtractionErrorCode.OVERWRITE_FILE: "File already exists",
        ExtractionErrorCode.LINK_TYPE: "Symlink or hardlink entry not permitted",
        ExtractionErrorCode.SPECIAL_TYPE: "Device, FIFO, or socket entry not permitted",
        ExtractionErrorCode.CASEFOLD_COLLISION: "Case-insensitive path collision detected",
        ExtractionErrorCode.DEPTH: "Path exceeds maximum depth",
        ExtractionErrorCode.SEGMENT_LEN: "Path component exceeds maximum length",
        ExtractionErrorCode.PATH_LEN: "Full path exceeds maximum length",
        ExtractionErrorCode.ENTRY_BUDGET: "Entry count exceeds maximum",
        ExtractionErrorCode.FILE_SIZE: "File size exceeds limit",
        ExtractionErrorCode.FILE_SIZE_STREAM: "Streamed file size exceeds limit",
        ExtractionErrorCode.ENTRY_RATIO: "Entry compression ratio exceeds limit",
        ExtractionErrorCode.SPACE: "Insufficient disk space",
        ExtractionErrorCode.FORMAT_NOT_ALLOWED: "Archive format or filter not in allow-list",
        ExtractionErrorCode.PORTABILITY: "Windows portability violation",
        ExtractionErrorCode.EXTRACT_CORRUPT: "Archive is corrupted or truncated",
        ExtractionErrorCode.EXTRACT_IO: "I/O error during extraction",
        ExtractionErrorCode.BOMB_RATIO: "Compression ratio indicates zip bomb",
    }
    msg = messages.get(code, str(code))
    if detail:
        msg += f": {detail}"
    return msg
