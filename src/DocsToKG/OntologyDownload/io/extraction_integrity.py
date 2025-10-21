# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_integrity",
#   "purpose": "Correctness & integrity verification for archive extraction",
#   "sections": [
#     {"id": "crc_integrity", "name": "CRC/Integrity Verification", "anchor": "CRC", "kind": "validators"},
#     {"id": "timestamps", "name": "Timestamp Policy", "anchor": "TIME", "kind": "policies"},
#     {"id": "unicode", "name": "Unicode Normalization", "anchor": "UNI", "kind": "validators"},
#     {"id": "format_allow", "name": "Format Allow-List", "anchor": "FMT", "kind": "validators"},
#     {"id": "ordering", "name": "Entry Ordering Determinism", "anchor": "ORDER", "kind": "policies"},
#     {"id": "duplicates", "name": "Duplicate Entry Handling", "anchor": "DUP", "kind": "policies"},
#     {"id": "manifest", "name": "Provenance Manifest", "anchor": "MANI", "kind": "outputs"}
#   ]
# }
# === /NAVMAP ===

"""Correctness & integrity verification for archive extraction.

Implements 6 correctness features + 2 cross-cutting integrity add-ons:
- CRC/integrity verification (per-entry & archive-level)
- Timestamp policy (deterministic & reproducible)
- Unicode normalization (cross-platform stability)
- Format allow-list (trusted formats only)
- Entry ordering determinism (stable outputs)
- Duplicate entry policy (unambiguous handling)
- Provenance manifest (auditable output)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import ConfigError
from .extraction_policy import ExtractionPolicy

# ============================================================================
# CRC / INTEGRITY VERIFICATION
# ============================================================================


# Windows reserved device names (case-insensitive)
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}


@dataclass
class IntegrityCheckResult:
    """Result of integrity verification for a single entry."""

    pathname: str
    size_declared: Optional[int]
    size_written: Optional[int]
    crc_declared: Optional[str]
    digest_computed: Optional[str]
    passed: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class IntegrityVerifier:
    """Verifies archive and entry-level integrity."""

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize integrity verifier.

        Args:
            policy: Extraction policy
        """
        self.policy = policy
        self.verify = policy.integrity_verify if hasattr(policy, "integrity_verify") else True
        self.fail_on_mismatch = (
            policy.integrity_fail_on_mismatch
            if hasattr(policy, "integrity_fail_on_mismatch")
            else True
        )
        self.checks: List[IntegrityCheckResult] = []
        self.entries_checked = 0
        self.crc_mismatches = 0
        self.size_mismatches = 0

    def check_entry(
        self,
        pathname: str,
        size_declared: Optional[int],
        size_written: Optional[int],
        crc_declared: Optional[str],
        digest_computed: Optional[str],
    ) -> IntegrityCheckResult:
        """Verify a single entry's integrity.

        Args:
            pathname: Entry pathname
            size_declared: Declared size from archive header
            size_written: Actual bytes written
            crc_declared: Declared CRC/hash (if present)
            digest_computed: Computed digest (if enabled)

        Returns:
            IntegrityCheckResult

        Raises:
            ConfigError: If verification fails and policy requires it
        """
        if not self.verify:
            return IntegrityCheckResult(
                pathname=pathname,
                size_declared=size_declared,
                size_written=size_written,
                crc_declared=crc_declared,
                digest_computed=digest_computed,
                passed=True,
            )

        self.entries_checked += 1
        result = IntegrityCheckResult(
            pathname=pathname,
            size_declared=size_declared,
            size_written=size_written,
            crc_declared=crc_declared,
            digest_computed=digest_computed,
            passed=True,
        )

        # Check size match
        if size_declared is not None and size_written is not None:
            if size_declared != size_written:
                result.passed = False
                result.error_code = "E_SIZE_MISMATCH"
                result.error_message = (
                    f"Size mismatch: declared {size_declared}, wrote {size_written}"
                )
                self.size_mismatches += 1

                if self.fail_on_mismatch:
                    raise ConfigError(result.error_message)

        # Check CRC/digest match
        if crc_declared and digest_computed:
            if crc_declared != digest_computed:
                result.passed = False
                result.error_code = "E_CRC_MISMATCH"
                result.error_message = (
                    f"CRC mismatch: declared {crc_declared}, computed {digest_computed}"
                )
                self.crc_mismatches += 1

                if self.fail_on_mismatch:
                    raise ConfigError(result.error_message)

        self.checks.append(result)
        return result


# ============================================================================
# TIMESTAMP POLICY
# ============================================================================


@dataclass
class TimestampPolicy:
    """Configuration for timestamp handling."""

    mode: str = "preserve"  # "preserve" | "normalize" | "source_date_epoch"
    normalize_to: str = "archive_mtime"  # "archive_mtime" | "now"
    preserve_dir_mtime: bool = False
    value: Optional[int] = None  # Used for normalize/source_date_epoch modes


def compute_target_mtime(
    entry_mtime: Optional[int],
    archive_mtime: int,
    timestamp_policy: TimestampPolicy,
) -> int:
    """Compute target mtime for an entry.

    Args:
        entry_mtime: Entry's mtime from archive (if available)
        archive_mtime: Archive file's mtime
        timestamp_policy: Timestamp policy

    Returns:
        Target mtime (seconds since epoch)
    """
    if timestamp_policy.mode == "preserve":
        return entry_mtime if entry_mtime is not None else archive_mtime
    elif timestamp_policy.mode == "normalize":
        if timestamp_policy.normalize_to == "archive_mtime":
            return archive_mtime
        else:  # "now"
            return int(time.time())
    elif timestamp_policy.mode == "source_date_epoch":
        # Read from environment or use stored value
        if timestamp_policy.value is not None:
            return timestamp_policy.value
        sde = os.environ.get("SOURCE_DATE_EPOCH")
        if sde:
            return int(sde)
        return int(time.time())
    else:
        return archive_mtime


def apply_mtime(path: Path, mtime: int) -> None:
    """Apply mtime to a file.

    Args:
        path: File path
        mtime: Modification time (seconds since epoch)
    """
    try:
        os.utime(str(path), (mtime, mtime))
    except OSError:
        pass  # Non-fatal


# ============================================================================
# UNICODE NORMALIZATION
# ============================================================================


def normalize_pathname(
    pathname: str,
    normalize_form: str = "NFC",
    on_decode_error: str = "reject",
) -> str:
    """Normalize pathname using Unicode form.

    Args:
        pathname: Original pathname
        normalize_form: Normalization form ("NFC" | "NFD")
        on_decode_error: Error handling ("reject" | "replace")

    Returns:
        Normalized pathname

    Raises:
        ConfigError: If normalization fails and reject mode is set
    """
    try:
        if normalize_form == "NFC":
            return unicodedata.normalize("NFC", pathname)
        elif normalize_form == "NFD":
            return unicodedata.normalize("NFD", pathname)
        else:
            return pathname
    except Exception as exc:
        if on_decode_error == "reject":
            raise ConfigError(f"Unicode normalization failed for {pathname}: {exc}") from exc
        else:  # "replace"
            # Replace invalid characters with U+FFFD
            return pathname.encode("utf-8", "replace").decode("utf-8")


# ============================================================================
# FORMAT ALLOW-LIST
# ============================================================================


def validate_format_allowed(
    format_name: str,
    filters: List[str],
    policy: ExtractionPolicy,
) -> None:
    """Validate archive format is in allow-list.

    Args:
        format_name: Detected archive format (e.g., "zip", "tar")
        filters: Detected compression filters (e.g., ["gzip"])
        policy: Extraction policy with allowed_formats/filters

    Raises:
        ConfigError: If format/filter not allowed
    """
    allowed_formats = getattr(policy, "allowed_formats", None)
    allowed_filters = getattr(policy, "allowed_filters", None)

    # Default allow-lists if not specified
    if allowed_formats is None:
        allowed_formats = ["zip", "tar", "ustar", "pax", "gnutar"]
    if allowed_filters is None:
        allowed_filters = ["gzip", "bzip2", "xz", "zstd", "none"]

    # Check format
    if format_name not in allowed_formats:
        raise ConfigError(f"Archive format not allowed: {format_name} (allowed: {allowed_formats})")

    # Check filters
    for filt in filters:
        if filt not in allowed_filters:
            raise ConfigError(
                f"Compression filter not allowed: {filt} (allowed: {allowed_filters})"
            )


# ============================================================================
# ENTRY ORDERING DETERMINISM
# ============================================================================


def get_sort_key(
    pathname: str,
    scan_index: int,
    order_mode: str,
) -> tuple:
    """Get sort key for entry based on ordering mode.

    Args:
        pathname: Entry pathname (normalized)
        scan_index: Sequential index from pre-scan
        order_mode: Ordering mode ("header" | "path_asc")

    Returns:
        Tuple usable as sort key
    """
    if order_mode == "header":
        return (scan_index,)
    elif order_mode == "path_asc":
        return (pathname,)
    else:
        return (scan_index,)


# ============================================================================
# DUPLICATE ENTRY POLICY
# ============================================================================


@dataclass
class DuplicateEntry:
    """Record of a duplicate entry."""

    pathname: str
    first_index: int
    duplicate_indices: List[int]
    policy_action: str  # "reject" | "first_wins" | "last_wins"
    result: str  # "rejected" | "skipped" | "replaced"


class DuplicateDetector:
    """Detects and handles duplicate entries."""

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize detector.

        Args:
            policy: Extraction policy
        """
        self.policy = policy
        self.duplicate_policy = getattr(policy, "duplicate_policy", "reject")
        self.seen_paths: Dict[str, int] = {}  # path -> first_index
        self.duplicates: List[DuplicateEntry] = []

    def check_entry(
        self,
        pathname: str,
        scan_index: int,
    ) -> tuple[bool, Optional[str]]:
        """Check if entry is duplicate.

        Args:
            pathname: Normalized pathname
            scan_index: Entry index from pre-scan

        Returns:
            (should_extract, action) tuple where:
            - should_extract: True if entry should be extracted
            - action: "extract" | "skip" | None

        Raises:
            ConfigError: If duplicate and reject policy
        """
        if pathname not in self.seen_paths:
            self.seen_paths[pathname] = scan_index
            return (True, "extract")

        # Duplicate detected
        first_index = self.seen_paths[pathname]

        if self.duplicate_policy == "reject":
            raise ConfigError(
                f"Duplicate entry detected: {pathname} (first at index {first_index}, "
                f"again at index {scan_index})"
            )
        elif self.duplicate_policy == "first_wins":
            self.duplicates.append(
                DuplicateEntry(
                    pathname=pathname,
                    first_index=first_index,
                    duplicate_indices=[scan_index],
                    policy_action="first_wins",
                    result="skipped",
                )
            )
            return (False, "skip")
        elif self.duplicate_policy == "last_wins":
            self.duplicates.append(
                DuplicateEntry(
                    pathname=pathname,
                    first_index=first_index,
                    duplicate_indices=[scan_index],
                    policy_action="last_wins",
                    result="replaced",
                )
            )
            # Update to new index (last wins)
            self.seen_paths[pathname] = scan_index
            return (True, "extract")

        return (True, "extract")


# ============================================================================
# PROVENANCE MANIFEST
# ============================================================================


@dataclass
class ManifestEntry:
    """Single entry in provenance manifest."""

    path_rel: str
    size: int
    sha256: Optional[str]
    mtime: int
    scan_index: int


@dataclass
class ProvenanceManifest:
    """Auditable record of extraction."""

    schema_version: str = "1.0"
    archive_path: str = ""
    archive_sha256: Optional[str] = None
    format: str = ""
    filters: List[str] = None
    timestamp_mode: str = "preserve"
    timestamp_value: Optional[int] = None
    policy: Dict = None
    entries: List[ManifestEntry] = None
    metrics: Dict = None
    manifest_sha256: Optional[str] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = []
        if self.entries is None:
            self.entries = []
        if self.policy is None:
            self.policy = {}
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "format": self.format,
            "filters": self.filters,
            "timestamp_mode": self.timestamp_mode,
            "timestamp_value": self.timestamp_value,
            "policy": self.policy,
            "entries": [
                {
                    "path_rel": e.path_rel,
                    "size": e.size,
                    "sha256": e.sha256,
                    "mtime": e.mtime,
                    "scan_index": e.scan_index,
                }
                for e in self.entries
            ],
            "metrics": self.metrics,
        }

    def to_json(self, include_digest: bool = False) -> str:
        """Serialize to JSON string.

        Args:
            include_digest: Whether to include manifest SHA256

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, sort_keys=True)

        if include_digest:
            # Compute manifest hash before storing it
            manifest_hash = hashlib.sha256(json_str.encode()).hexdigest()
            data["manifest_sha256"] = manifest_hash
            json_str = json.dumps(data, indent=2, sort_keys=True)

        return json_str

    def write_to_file(self, path: Path, include_digest: bool = True) -> None:
        """Write manifest to file atomically.

        Args:
            path: Target file path
            include_digest: Whether to include manifest SHA256
        """
        json_str = self.to_json(include_digest=include_digest)

        # Write atomically via temp + rename
        temp_path = path.parent / f".{path.name}.tmp-{os.getpid()}"
        try:
            temp_path.write_text(json_str)
            os.replace(str(temp_path), str(path))
        except Exception:
            # Non-fatal
            if temp_path.exists():
                temp_path.unlink()


def check_windows_portability(path: str, policy: ExtractionPolicy) -> tuple[bool, Optional[str]]:
    """Check if a path violates Windows portability rules.

    Args:
        path: Path to check
        policy: Extraction policy with Windows portability setting

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not policy.windows_portability_strict:
        return True, None

    # Extract filename from path (last component)
    parts = path.rstrip("/\\").split("/")
    if not parts:
        return False, "Empty path"

    filename = parts[-1]
    if not filename:
        return False, "Empty filename"

    # Check for trailing space or dot
    if filename != filename.rstrip(". "):
        return False, f"Filename has trailing space or dot: {filename}"

    # Check for Windows reserved names
    # Handle cases like "CON.txt" which are also reserved
    name_part = filename.split(".")[0].lower()
    if name_part in _WINDOWS_RESERVED_NAMES:
        return False, f"Filename is a Windows reserved name: {filename}"

    return True, None


def validate_archive_format(
    format_name: Optional[str],
    filters: Optional[List[str]],
    policy: ExtractionPolicy,
) -> tuple[bool, Optional[str]]:
    """Validate archive format and compression filters against allow-list.

    Args:
        format_name: Detected archive format name
        filters: List of compression filters applied
        policy: Extraction policy with allowed formats/filters

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get defaults if not specified
    allowed_formats = policy.allowed_formats or {
        "zip",
        "tar",
        "ustar",
        "pax",
        "gnutar",
        "7z",
        "iso9660",
        "cpio",
    }
    allowed_filters = policy.allowed_filters or {
        "none",
        "gzip",
        "bzip2",
        "xz",
        "zstd",
        "lz4",
        "compress",
    }

    # Validate format (check if any allowed format is contained in the detected format name)
    if format_name:
        format_lower = format_name.lower()
        found_match = False
        for allowed_format in allowed_formats:
            if allowed_format.lower() in format_lower:
                found_match = True
                break

        if not found_match:
            return (
                False,
                f"Format '{format_name}' not in allow-list: {', '.join(sorted(allowed_formats))}",
            )

    # Validate filters
    if filters:
        for filter_name in filters:
            if filter_name and filter_name.lower() not in {f.lower() for f in allowed_filters}:
                return (
                    False,
                    f"Filter '{filter_name}' not in allow-list: {', '.join(sorted(allowed_filters))}",
                )

    return True, None
