# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_constraints",
#   "purpose": "Constraint validators for archive extraction security policies (Phase 2)",
#   "sections": [
#     {"id": "path_validation", "name": "Path Normalization & Validation", "anchor": "PATH", "kind": "validators"},
#     {"id": "type_validation", "name": "Entry Type Validation", "anchor": "TYPE", "kind": "validators"},
#     {"id": "constraint_validation", "name": "Constraint Enforcement", "anchor": "CONSTR", "kind": "validators"},
#     {"id": "collision_detection", "name": "Collision Detection", "anchor": "COLL", "kind": "validators"}
#   ]
# }
# === /NAVMAP ===

"""Constraint validators for Phase 2 archive extraction security hardening.

Implements validators for:
- Link and hardlink defense-in-depth checks
- Device/FIFO/socket quarantine
- Path normalization and constraints (depth, length, unicode)
- Case-fold collision detection
"""

from __future__ import annotations

import unicodedata
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set

from ..errors import ConfigError
from .extraction_policy import ExtractionPolicy
from .extraction_telemetry import ExtractionErrorCode, error_message


# ============================================================================
# PATH NORMALIZATION & VALIDATION
# ============================================================================

def normalize_path_unicode(path: str, policy: ExtractionPolicy) -> str:
    """Normalize path using the configured Unicode normalization form.
    
    Args:
        path: Path string to normalize
        policy: Extraction policy with unicode normalization setting
    
    Returns:
        Normalized path string
    
    Raises:
        ConfigError: If normalization setting is invalid
    """
    if policy.normalize_unicode == "none":
        return path
    
    if policy.normalize_unicode not in ("NFC", "NFD"):
        raise ConfigError(f"Invalid unicode normalization: {policy.normalize_unicode}")
    
    return unicodedata.normalize(policy.normalize_unicode, path)


def validate_path_constraints(
    path: str,
    policy: ExtractionPolicy,
) -> None:
    """Validate path against depth, component length, and total length constraints.
    
    Args:
        path: Path string to validate
        policy: Extraction policy with constraint limits
    
    Raises:
        ConfigError: If any constraint is violated
    """
    # Normalize first
    normalized = normalize_path_unicode(path, policy)
    
    # Split into components
    parts = PurePosixPath(normalized).parts
    
    # Check depth
    if len(parts) > policy.max_depth:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.DEPTH,
                f"Path has {len(parts)} components, max is {policy.max_depth}: {path}",
            )
        )
    
    # Check each component length (in UTF-8 bytes)
    for part in parts:
        part_bytes = part.encode("utf-8")
        if len(part_bytes) > policy.max_components_len:
            raise ConfigError(
                error_message(
                    ExtractionErrorCode.SEGMENT_LEN,
                    f"Component '{part}' is {len(part_bytes)} bytes, max is {policy.max_components_len}",
                )
            )
    
    # Check total path length (in UTF-8 bytes)
    path_bytes = normalized.encode("utf-8")
    if len(path_bytes) > policy.max_path_len:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.PATH_LEN,
                f"Full path is {len(path_bytes)} bytes, max is {policy.max_path_len}",
            )
        )


# ============================================================================
# ENTRY TYPE VALIDATION
# ============================================================================

def validate_entry_type(
    *,
    is_symlink: bool,
    is_hardlink: bool,
    is_fifo: bool,
    is_block_dev: bool,
    is_char_dev: bool,
    is_socket: bool,
    policy: ExtractionPolicy,
    entry_path: str,
) -> None:
    """Validate that entry type is permitted by policy.
    
    Args:
        is_symlink: Whether entry is a symlink
        is_hardlink: Whether entry is a hardlink
        is_fifo: Whether entry is a FIFO/named pipe
        is_block_dev: Whether entry is a block device
        is_char_dev: Whether entry is a character device
        is_socket: Whether entry is a socket
        policy: Extraction policy
        entry_path: Original entry path for error messages
    
    Raises:
        ConfigError: If entry type is not permitted
    """
    if not policy.allow_symlinks and is_symlink:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.LINK_TYPE,
                f"Symlink not permitted: {entry_path}",
            )
        )
    
    if not policy.allow_hardlinks and is_hardlink:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.LINK_TYPE,
                f"Hardlink not permitted: {entry_path}",
            )
        )
    
    if is_fifo or is_block_dev or is_char_dev or is_socket:
        entry_type = "FIFO" if is_fifo else "device" if (is_block_dev or is_char_dev) else "socket"
        raise ConfigError(
            error_message(
                ExtractionErrorCode.SPECIAL_TYPE,
                f"{entry_type} not permitted: {entry_path}",
            )
        )


# ============================================================================
# COLLISION DETECTION (CASE-FOLD)
# ============================================================================

class CaseCollisionDetector:
    """Detects case-fold collisions in archive member paths.
    
    Maintains a set of casefolded paths seen during extraction and detects
    when new entries would collide after casefolding normalization.
    """

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize detector with policy.
        
        Args:
            policy: Extraction policy (determines collision handling)
        """
        self.policy = policy
        self.casefolded: Set[str] = set()

    def check_collision(self, path: str) -> None:
        """Check if path collides with previously seen paths after casefolding.
        
        Args:
            path: Normalized path to check
        
        Raises:
            ConfigError: If collision detected and policy is "reject"
        """
        if self.policy.casefold_collision_policy == "allow":
            # Collision detection is disabled
            self.casefolded.add(path.casefold())
            return
        
        casefolded = path.casefold()
        if casefolded in self.casefolded:
            raise ConfigError(
                error_message(
                    ExtractionErrorCode.CASEFOLD_COLLISION,
                    f"Case-insensitive collision detected: {path}",
                )
            )
        self.casefolded.add(casefolded)


# ============================================================================
# ENTRY BUDGET VALIDATION
# ============================================================================

def validate_entry_count(
    current_count: int,
    policy: ExtractionPolicy,
) -> None:
    """Validate that entry count hasn't exceeded the budget.
    
    Args:
        current_count: Current number of entries processed
        policy: Extraction policy with max_entries limit
    
    Raises:
        ConfigError: If entry count exceeds limit
    """
    if current_count > policy.max_entries:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.ENTRY_BUDGET,
                f"Entry count {current_count} exceeds limit {policy.max_entries}",
            )
        )


# ============================================================================
# PER-FILE SIZE VALIDATION
# ============================================================================

def validate_file_size(
    declared_size: Optional[int],
    policy: ExtractionPolicy,
    entry_path: str,
) -> None:
    """Validate that file's declared size is within limits.
    
    Args:
        declared_size: Size reported by archive (may be None/unknown)
        policy: Extraction policy with max_file_size_bytes limit
        entry_path: Original entry path for error messages
    
    Raises:
        ConfigError: If declared size exceeds limit
    """
    if declared_size is None:
        # Size unknown; will be checked during streaming
        return
    
    if declared_size > policy.max_file_size_bytes:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.FILE_SIZE,
                f"File {entry_path} declares {declared_size} bytes, max is {policy.max_file_size_bytes}",
            )
        )


def validate_streaming_file_size(
    bytes_written: int,
    policy: ExtractionPolicy,
    entry_path: str,
) -> None:
    """Validate that streamed file size doesn't exceed limit during extraction.
    
    This is called periodically during file streaming to enforce the limit
    before the entire file is written.
    
    Args:
        bytes_written: Bytes written so far
        policy: Extraction policy with max_file_size_bytes limit
        entry_path: Original entry path for error messages
    
    Raises:
        ConfigError: If streamed size exceeds limit
    """
    if bytes_written > policy.max_file_size_bytes:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.FILE_SIZE_STREAM,
                f"File {entry_path} exceeded {policy.max_file_size_bytes} bytes during streaming",
            )
        )


# ============================================================================
# PER-ENTRY COMPRESSION RATIO VALIDATION
# ============================================================================

def validate_entry_compression_ratio(
    uncompressed_size: int,
    compressed_size: int,
    policy: ExtractionPolicy,
    entry_path: str,
) -> None:
    """Validate that entry's compression ratio is within limits.
    
    Args:
        uncompressed_size: Declared uncompressed size
        compressed_size: Actual compressed size in archive
        policy: Extraction policy with max_entry_ratio limit
        entry_path: Original entry path for error messages
    
    Raises:
        ConfigError: If compression ratio exceeds limit
    """
    if compressed_size <= 0:
        # Can't compute ratio
        return
    
    ratio = uncompressed_size / float(compressed_size)
    if ratio > policy.max_entry_ratio:
        raise ConfigError(
            error_message(
                ExtractionErrorCode.ENTRY_RATIO,
                f"File {entry_path} has compression ratio {ratio:.1f}:1, max is {policy.max_entry_ratio}:1",
            )
        )


# ============================================================================
# BATCH VALIDATORS (Convenience functions)
# ============================================================================

class PreScanValidator:
    """Orchestrates all pre-scan validations for a single archive entry.
    
    Groups related validations to reduce code duplication in filesystem.py.
    """

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize validator with policy.
        
        Args:
            policy: Extraction policy defining all constraints
        """
        self.policy = policy
        self.collision_detector = CaseCollisionDetector(policy)
        self.entry_count = 0

    def validate_entry(
        self,
        *,
        original_path: str,
        is_dir: bool,
        is_symlink: bool,
        is_hardlink: bool,
        is_fifo: bool,
        is_block_dev: bool,
        is_char_dev: bool,
        is_socket: bool,
        uncompressed_size: Optional[int] = None,
        compressed_size: Optional[int] = None,
    ) -> None:
        """Validate a single archive entry against all Phase 2 policies.
        
        Args:
            original_path: Original pathname from archive
            is_dir: Whether entry is a directory
            is_symlink: Whether entry is a symlink
            is_hardlink: Whether entry is a hardlink
            is_fifo: Whether entry is a FIFO
            is_block_dev: Whether entry is a block device
            is_char_dev: Whether entry is a character device
            is_socket: Whether entry is a socket
            uncompressed_size: Declared uncompressed size (if known)
            compressed_size: Compressed size (for ratio calculation)
        
        Raises:
            ConfigError: If any constraint is violated
        """
        # 1. Count and check entry budget
        self.entry_count += 1
        validate_entry_count(self.entry_count, self.policy)
        
        # 2. Validate entry type (symlink, hardlink, device, FIFO, socket)
        validate_entry_type(
            is_symlink=is_symlink,
            is_hardlink=is_hardlink,
            is_fifo=is_fifo,
            is_block_dev=is_block_dev,
            is_char_dev=is_char_dev,
            is_socket=is_socket,
            policy=self.policy,
            entry_path=original_path,
        )
        
        # 3. Normalize and validate path constraints
        normalized = normalize_path_unicode(original_path, self.policy)
        validate_path_constraints(normalized, self.policy)
        
        # 4. Check for case-fold collisions
        self.collision_detector.check_collision(normalized)
        
        # 5. Validate per-file size limits (for non-directories)
        if not is_dir and uncompressed_size is not None:
            validate_file_size(uncompressed_size, self.policy, original_path)
        
        # 6. Validate per-entry compression ratio (if sizes available)
        if not is_dir and compressed_size is not None and compressed_size > 0:
            if uncompressed_size is not None:
                validate_entry_compression_ratio(
                    uncompressed_size,
                    compressed_size,
                    self.policy,
                    original_path,
                )
