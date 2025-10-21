"""Garbage collection and retention utilities for the artifact catalog.

Provides tools for:
  - Finding orphaned files (in storage but not in catalog)
  - Age-based retention filtering
  - Safe deletion of unreferenced files with dry-run mode
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set

from DocsToKG.ContentDownload.catalog.models import DocumentRecord

logger = logging.getLogger(__name__)


def find_orphans(
    root_dir: str,
    referenced_paths: Set[str],
    follow_symlinks: bool = False,
) -> List[str]:
    """Find orphaned files in storage root not referenced by catalog.
    
    Walks the entire root_dir and identifies files that are not in the
    set of referenced catalog paths. These are candidates for deletion.
    
    Args:
        root_dir: Storage root directory to scan
        referenced_paths: Set of paths referenced in catalog (absolute paths)
        follow_symlinks: If True, follow symbolic links when walking
        
    Returns:
        List of absolute paths to orphaned files
    """
    orphans: List[str] = []
    root = Path(root_dir)
    
    if not root.exists():
        logger.warning(f"Root directory does not exist: {root_dir}")
        return orphans
    
    try:
        for fpath in root.rglob("*"):
            if fpath.is_file():
                if str(fpath) not in referenced_paths:
                    orphans.append(str(fpath))
    except Exception as e:
        logger.error(f"Error scanning root directory: {e}")
    
    logger.info(f"Found {len(orphans)} orphaned files in {root_dir}")
    return orphans


def retention_filter(
    records: List[DocumentRecord],
    retention_days: int,
) -> List[DocumentRecord]:
    """Filter records older than retention policy.
    
    Args:
        records: List of document records
        retention_days: Age threshold in days (records older than this are returned)
        
    Returns:
        List of records older than retention_days
    """
    if retention_days <= 0:
        return []
    
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    expired = [r for r in records if r.created_at < cutoff]
    
    logger.info(f"Retention filter: {len(expired)} records older than {retention_days} days")
    return expired


def collect_referenced_paths(
    records: List[DocumentRecord],
) -> Set[str]:
    """Extract all referenced storage URIs from catalog records.
    
    Args:
        records: List of document records
        
    Returns:
        Set of storage URIs (file:// paths converted to local paths)
    """
    paths = set()
    
    for record in records:
        # Convert file:// URIs to local paths
        uri = record.storage_uri
        if uri.startswith("file://"):
            path = uri[7:]  # Remove "file://" prefix
            # Normalize path
            path = str(Path(path).resolve())
            paths.add(path)
        elif uri.startswith("s3://"):
            # S3 URIs are not local files (skip)
            pass
        else:
            # Assume it's a local path
            path = str(Path(uri).resolve())
            paths.add(path)
    
    logger.debug(f"Collected {len(paths)} referenced paths")
    return paths


def delete_orphan_files(
    orphan_paths: List[str],
    dry_run: bool = True,
) -> int:
    """Delete orphaned files with optional dry-run mode.
    
    Args:
        orphan_paths: List of file paths to delete
        dry_run: If True, log but don't delete
        
    Returns:
        Number of files deleted (or would be deleted in dry-run)
    """
    deleted_count = 0
    
    for path_str in orphan_paths:
        path = Path(path_str)
        
        if not path.exists():
            logger.debug(f"File already gone: {path}")
            deleted_count += 1
            continue
        
        if not path.is_file():
            logger.warning(f"Not a file (skipped): {path}")
            continue
        
        try:
            if dry_run:
                logger.info(f"[DRY-RUN] Would delete: {path} ({path.stat().st_size} bytes)")
            else:
                path.unlink()
                logger.info(f"Deleted: {path}")
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
    
    action = "would delete" if dry_run else "deleted"
    logger.info(f"GC {action} {deleted_count}/{len(orphan_paths)} orphaned files")
    return deleted_count


class RetentionPolicy:
    """Encapsulates retention and GC policies."""
    
    def __init__(
        self,
        retention_days: int = 0,
        orphan_ttl_days: int = 7,
    ):
        """Initialize retention policy.
        
        Args:
            retention_days: Age in days for record retention (0 = disabled)
            orphan_ttl_days: Age in days for orphan file eligibility
        """
        self.retention_days = retention_days
        self.orphan_ttl_days = orphan_ttl_days
    
    def should_retain_record(self, record: DocumentRecord) -> bool:
        """Check if a record should be retained.
        
        Args:
            record: Document record to check
            
        Returns:
            True if record should be kept, False if it should be deleted
        """
        if self.retention_days <= 0:
            return True
        
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        return record.created_at >= cutoff
    
    def should_gc_file(self, file_mtime: datetime) -> bool:
        """Check if a file should be garbage collected.
        
        Args:
            file_mtime: File modification time
            
        Returns:
            True if file is older than orphan_ttl and should be deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=self.orphan_ttl_days)
        return file_mtime < cutoff
