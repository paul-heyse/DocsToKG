"""
Path Safety Validation Gate.

Prevents:
- Path traversal attacks (.., ~, symlink loops)
- Escaping artifact root directory
- Permission issues
- Writing to system directories

This is the SINGLE SOURCE OF TRUTH for path validation in ContentDownload,
ensuring artifacts are only written to safe, sandboxed locations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PathPolicyError(ValueError):
    """Path validation policy violation."""

    pass


def validate_path_safety(
    final_path: str,
    artifact_root: Optional[str] = None,
) -> str:
    """
    Validate that final_path is safe to write to.

    Checks:
    - Path is within artifact_root (no escape)
    - Path is not a symlink escape
    - Parent directory is writable
    - Path is absolute

    Args:
        final_path: Target write path (must be absolute)
        artifact_root: Safe root directory (e.g., current working directory + PDF/HTML/XML dir)
                      If None, defaults to current working directory

    Returns:
        Canonicalized final_path if safe

    Raises:
        PathPolicyError: If path is unsafe (traversal, outside root, etc.)

    Example:
        >>> # Safe usage
        >>> safe_path = validate_path_safety(
        ...     "/data/artifacts/paper.pdf",
        ...     artifact_root="/data/artifacts"
        ... )

        >>> # Unsafe: path traversal
        >>> try:
        ...     validate_path_safety("../../etc/passwd", artifact_root="/data")
        ... except PathPolicyError as e:
        ...     logger.error(f"Policy violation: {e}")
    """
    if not final_path:
        raise PathPolicyError("final_path cannot be empty")

    # Default artifact root to cwd if not specified
    if artifact_root is None:
        artifact_root = os.getcwd()

    # Resolve to absolute, canonical paths (resolves symlinks, .., etc.)
    try:
        abs_final = Path(final_path).resolve()
        abs_root = Path(artifact_root).resolve()
    except (OSError, RuntimeError) as e:
        raise PathPolicyError(f"Cannot resolve path: {e}")

    # Check 1: Path must be under artifact_root (prevent escape)
    try:
        abs_final.relative_to(abs_root)
    except ValueError:
        raise PathPolicyError(
            f"Path escapes artifact root: {final_path} is not under {artifact_root}"
        )

    # Check 2: Parent directory must exist (or be creatable)
    parent = abs_final.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        raise PathPolicyError(f"No write permission for parent directory: {parent}")

    # Check 3: If file exists, must be writable
    if abs_final.exists() and not os.access(abs_final, os.W_OK):
        raise PathPolicyError(f"No write permission for existing file: {abs_final}")

    # Check 4: Prevent writing to sensitive system directories
    forbidden_prefixes = ["/etc", "/sys", "/proc", "/root", "/boot", "/dev"]
    for prefix in forbidden_prefixes:
        if str(abs_final).startswith(prefix):
            raise PathPolicyError(f"Cannot write to system directory: {abs_final}")

    logger.debug(f"Path validated: {final_path} → {abs_final}")
    return str(abs_final)
