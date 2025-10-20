# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.OntologyDownload.io.extraction_throughput",
#   "purpose": "Throughput & I/O efficiency optimizations for archive extraction",
#   "sections": [
#     {"id": "adaptive_buffer", "name": "Adaptive Buffer Sizing", "anchor": "BUFFER", "kind": "optimizers"},
#     {"id": "preallocation", "name": "File Preallocation", "anchor": "PREALLOC", "kind": "optimizers"},
#     {"id": "atomic_writes", "name": "Atomic Writes & Fsync", "anchor": "ATOMIC", "kind": "optimizers"},
#     {"id": "hashing", "name": "Inline Hashing Pipeline", "anchor": "HASH", "kind": "optimizers"},
#     {"id": "selective", "name": "Selective Extraction", "anchor": "SELECT", "kind": "optimizers"},
#     {"id": "cpu_guard", "name": "CPU Guard (Wall-Time Limit)", "anchor": "CPU", "kind": "optimizers"}
#   ]
# }
# === /NAVMAP ===

"""Throughput & I/O efficiency optimizations for archive extraction.

Implements 8 performance optimizations:
- Adaptive buffer sizing (match workload)
- File preallocation (reduce fragmentation)
- Atomic per-file writes (fsync discipline)
- Inline hashing (content integrity)
- Selective extraction (skip unnecessary files)
- CPU guard (wall-time limits)
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from ..errors import ConfigError
from .extraction_policy import ExtractionPolicy


# ============================================================================
# ADAPTIVE BUFFER SIZING
# ============================================================================


def compute_adaptive_buffer_size(
    entries: list,
    policy: ExtractionPolicy,
) -> int:
    """Compute optimal copy buffer size based on file size distribution.

    Analyzes the archive's entry size distribution and chooses a buffer
    size that balances syscall overhead and memory use.

    Args:
        entries: List of archive entries with size metadata
        policy: Extraction policy with buffer min/max

    Returns:
        Optimal buffer size in bytes
    """
    if not entries:
        return 256 * 1024  # 256 KiB default

    # Compute size distribution
    small_count = 0  # < 64 KiB
    medium_count = 0  # < 4 MiB
    large_count = 0  # >= 4 MiB

    small_bytes = 0
    medium_bytes = 0
    large_bytes = 0

    for size in entries:
        if size is None:
            continue
        if size < 64 * 1024:
            small_count += 1
            small_bytes += size
        elif size < 4 * 1024 * 1024:
            medium_count += 1
            medium_bytes += size
        else:
            large_count += 1
            large_bytes += size

    total_bytes = small_bytes + medium_bytes + large_bytes
    if total_bytes == 0:
        return 256 * 1024

    # Heuristic: choose buffer size based on data distribution
    large_pct = large_bytes / total_bytes
    small_pct = small_bytes / total_bytes

    if large_pct > 0.5:
        # Mostly large files: use larger buffer
        return min(1024 * 1024, policy.copy_buffer_max)
    elif small_pct > 0.5:
        # Mostly small files: use smaller buffer
        return max(64 * 1024, policy.copy_buffer_min)
    else:
        # Mixed: use middle ground
        return 256 * 1024


# ============================================================================
# FILE PREALLOCATION
# ============================================================================


def preallocate_file(
    path: Path,
    size: int,
    policy: ExtractionPolicy,
) -> None:
    """Preallocate file to reduce fragmentation and ENOSPC risk.

    Uses `posix_fallocate` when available; falls back gracefully.

    Args:
        path: File path to preallocate
        size: Number of bytes to preallocate
        policy: Extraction policy

    Raises:
        ConfigError: If preallocation fails and policy requires it
    """
    if not policy.preallocate or size == 0:
        return

    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            # Try posix_fallocate (Unix)
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(fd, 0, size)
            else:
                # Fallback: seek and write a sentinel byte
                os.lseek(fd, size - 1, os.SEEK_SET)
                os.write(fd, b"\x00")
        finally:
            os.close(fd)
    except OSError as exc:
        raise ConfigError(
            f"Preallocation failed for {path}: {size} bytes, errno={exc.errno}"
        ) from exc


# ============================================================================
# ATOMIC WRITES & FSYNC DISCIPLINE
# ============================================================================


def create_temp_path(parent_dir: Path, base_name: str) -> Path:
    """Create a temporary file path for atomic writes.

    Args:
        parent_dir: Directory for the temporary file
        base_name: Base name of the final file

    Returns:
        Temporary file path with unique suffix
    """
    # Format: .base_name.tmp-pid-counter
    counter = int(time.time() * 1000000) % 1000000
    return parent_dir / f".{base_name}.tmp-{os.getpid()}-{counter}"


def atomic_rename_and_fsync(
    temp_path: Path,
    final_path: Path,
    parent_dir: Path,
    policy: ExtractionPolicy,
    rename_count: int,
) -> int:
    """Atomically rename temp file and optionally fsync parent directory.

    Args:
        temp_path: Temporary file path
        final_path: Final destination path
        parent_dir: Parent directory path
        policy: Extraction policy
        rename_count: Number of renames done so far

    Returns:
        Updated rename count
    """
    # Fsync the file first
    if policy.atomic:
        try:
            fd = os.open(str(temp_path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # Non-fatal

    # Rename atomically
    os.replace(str(temp_path), str(final_path))
    rename_count += 1

    # Fsync parent directory periodically
    if policy.atomic and rename_count % policy.group_fsync == 0:
        try:
            parent_fd = os.open(str(parent_dir), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass  # Non-fatal

    return rename_count


# ============================================================================
# INLINE HASHING PIPELINE
# ============================================================================


class HashingPipeline:
    """Computes content hashes during extraction."""

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize hashing pipeline.

        Args:
            policy: Extraction policy with hash settings
        """
        self.policy = policy
        self.enabled = policy.hash_enable
        self.algorithms = policy.hash_algorithms or ["sha256"]
        self.file_digests: dict[str, dict[str, str]] = {}

    def start_file(self, path_key: str) -> None:
        """Start hashing a new file.

        Args:
            path_key: Unique key for the file
        """
        if not self.enabled:
            return

        self.file_digests[path_key] = {
            algo: hashlib.new(algo).hexdigest() for algo in self.algorithms
        }

    def update(self, path_key: str, data: bytes) -> None:
        """Update hashes with new data.

        Args:
            path_key: File key
            data: Bytes to hash
        """
        if not self.enabled or path_key not in self.file_digests:
            return

        for algo in self.algorithms:
            hasher = hashlib.new(algo)
            hasher.update(data)
            self.file_digests[path_key][algo] = hasher.hexdigest()

    def get_digests(self, path_key: str) -> dict[str, str]:
        """Get final digests for a file.

        Args:
            path_key: File key

        Returns:
            Dictionary mapping algorithm names to hex digests
        """
        return self.file_digests.get(path_key, {})


# ============================================================================
# SELECTIVE EXTRACTION (INCLUDE/EXCLUDE PATTERNS)
# ============================================================================


def should_extract_entry(
    pathname: str,
    policy: ExtractionPolicy,
) -> bool:
    """Determine if an entry should be extracted based on globs.

    Args:
        pathname: Entry pathname (normalized)
        policy: Extraction policy with include/exclude globs

    Returns:
        True if entry should be extracted, False otherwise
    """
    # If no globs specified, include everything
    if not policy.include_globs and not policy.exclude_globs:
        return True

    # Check exclude list first (most restrictive)
    if policy.exclude_globs:
        for pattern in policy.exclude_globs:
            if fnmatch.fnmatch(pathname, pattern):
                return False

    # Check include list (if specified, must match)
    if policy.include_globs:
        for pattern in policy.include_globs:
            if fnmatch.fnmatch(pathname, pattern):
                return True
        # If include list is specified but path didn't match, exclude it
        return False

    # No include list specified and didn't match exclude → include
    return True


# ============================================================================
# CPU GUARD (WALL-TIME LIMIT)
# ============================================================================


class CPUGuard:
    """Enforces wall-time limits on extraction."""

    def __init__(self, policy: ExtractionPolicy) -> None:
        """Initialize CPU guard.

        Args:
            policy: Extraction policy with max_wall_time_seconds
        """
        self.policy = policy
        self.start_time = time.time()
        self.check_count = 0

    def check(self) -> None:
        """Check if wall-time limit exceeded.

        Raises:
            ConfigError: If time limit exceeded and action is "abort"
        """
        self.check_count += 1

        # Only check every 100 calls to minimize overhead
        if self.check_count % 100 != 0:
            return

        elapsed = time.time() - self.start_time

        if elapsed > self.policy.max_wall_time_seconds:
            if self.policy.cpu_guard_action == "abort":
                raise ConfigError(
                    f"Extraction wall-time exceeded: {elapsed:.1f}s > {self.policy.max_wall_time_seconds}s"
                )
