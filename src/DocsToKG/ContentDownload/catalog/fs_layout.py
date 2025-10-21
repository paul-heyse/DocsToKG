"""Filesystem-based storage layout strategies for artifact catalog.

Provides two storage layout options:
  1. Policy path: human-friendly paths derived from artifact metadata
  2. CAS (Content-Addressable Storage): paths based on SHA-256 hash
  
Also handles deduplication via hardlinks or copies.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def cas_path(root_dir: str, sha256_hex: str) -> str:
    """Generate CAS path from SHA-256 hash.
    
    Uses two-level directory fan-out to avoid hot directories.
    
    Example:
        cas_path("/data", "e3b0c44298fc1c14...") 
        -> "/data/cas/sha256/e3/b0c44298fc1c14..."
    
    Args:
        root_dir: Base root directory
        sha256_hex: SHA-256 hash in lowercase hex format
        
    Returns:
        Full path to CAS file location
        
    Raises:
        ValueError: If sha256_hex is invalid
    """
    if not sha256_hex or len(sha256_hex) < 4:
        raise ValueError(f"Invalid SHA-256: {sha256_hex}")
    
    path = Path(root_dir) / "cas" / "sha256" / sha256_hex[:2] / sha256_hex[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def policy_path(root_dir: str, *, artifact_id: str, url_basename: str) -> str:
    """Generate human-friendly policy path from artifact metadata.
    
    Creates predictable, browsable paths based on artifact metadata.
    
    Example:
        policy_path("/data", artifact_id="doi:10.1234/abc", url_basename="paper.pdf")
        -> "/data/paper.pdf"
    
    Args:
        root_dir: Base root directory
        artifact_id: Artifact identifier (unused in basic implementation)
        url_basename: Basename from URL (e.g., filename)
        
    Returns:
        Full path to policy file location
    """
    path = Path(root_dir) / url_basename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def dedup_hardlink_or_copy(
    src_tmp: str,
    dst_final: str,
    hardlink: bool = True,
    verify_inode: bool = False,
) -> bool:
    """Deduplicate via hardlink or copy.
    
    Attempts hardlink if dst_final exists and hardlink=True. On success,
    removes src_tmp and returns True. Falls back to move/copy on error.
    
    Args:
        src_tmp: Source temporary file path
        dst_final: Destination final file path
        hardlink: If True, attempt hardlink for dedup
        verify_inode: If True, verify inode equality after hardlink
        
    Returns:
        True if dedup was successful (hardlink hit or copy)
        False if file didn't exist (new file)
        
    Raises:
        OSError: If final operations fail
    """
    src = Path(src_tmp)
    dst = Path(dst_final)
    
    # If destination doesn't exist, this is a new file (no dedup)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_tmp, dst_final)
        logger.debug(f"New file: {dst_final}")
        return False
    
    # Destination exists - this is a dedup hit
    logger.info(f"Dedup hit detected: {dst_final}")
    
    if hardlink:
        try:
            # Try hardlink if destination exists
            os.link(dst_final, src_tmp)
            # Now replace destination with the linked version
            os.remove(dst_final)
            os.replace(src_tmp, dst_final)
            
            if verify_inode:
                src_stat = os.stat(src_tmp)
                dst_stat = os.stat(dst_final)
                if src_stat.st_ino == dst_stat.st_ino:
                    logger.debug(f"Hardlink verified: inode {src_stat.st_ino}")
                else:
                    logger.warning(f"Hardlink inode mismatch: {src_stat.st_ino} vs {dst_stat.st_ino}")
            
            logger.info(f"Hardlink dedup successful: {dst_final}")
            return True
        except OSError as e:
            logger.debug(f"Hardlink failed: {e}, falling back to copy")
            # Fall through to copy
    
    # Fallback: copy to destination (may be same filesystem or cross-filesystem)
    try:
        if dst.exists():
            os.remove(dst_final)
        shutil.copy2(src_tmp, dst_final)
        os.remove(src_tmp)
        logger.info(f"Copy dedup successful: {dst_final}")
        return True
    except OSError as e:
        logger.error(f"Failed to deduplicate: {e}")
        raise


def choose_final_path(
    root_dir: str,
    layout: str,
    sha256_hex: Optional[str],
    artifact_id: str,
    url_basename: str,
) -> str:
    """Choose final path based on layout strategy and availability.
    
    Args:
        root_dir: Root storage directory
        layout: Layout strategy ('policy_path' or 'cas')
        sha256_hex: SHA-256 hash (required for CAS layout)
        artifact_id: Artifact identifier
        url_basename: URL basename
        
    Returns:
        Final path to use for storage
        
    Raises:
        ValueError: If CAS layout requested but sha256 unavailable
    """
    if layout == "cas":
        if not sha256_hex:
            raise ValueError("CAS layout requires sha256_hex")
        return cas_path(root_dir, sha256_hex)
    elif layout == "policy_path":
        return policy_path(root_dir, artifact_id=artifact_id, url_basename=url_basename)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def extract_basename_from_url(url: str) -> str:
    """Extract filename from URL, with fallback.
    
    Args:
        url: Source URL
        
    Returns:
        Filename or fallback identifier
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        basename = Path(parsed.path).name
        if basename:
            return basename
    except Exception:
        pass
    
    # Fallback: use last part of domain + hash
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"artifact_{url_hash}.bin"
