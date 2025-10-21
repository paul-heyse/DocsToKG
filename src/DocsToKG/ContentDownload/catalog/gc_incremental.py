"""Incremental garbage collection with batching and progress tracking.

Provides production-safe GC with:
  - Configurable batch size
  - Pause/resume capability
  - Progress reporting
  - Atomic per-batch operations
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from DocsToKG.ContentDownload.catalog.gc import (
    collect_referenced_paths,
    find_orphans,
)
from DocsToKG.ContentDownload.catalog.store import CatalogStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GCBatchResult:
    """Result of a single GC batch."""
    batch_num: int
    files_processed: int
    files_deleted: int
    errors: int
    elapsed_seconds: float
    total_freed_bytes: int


@dataclass(frozen=True)
class GCStats:
    """Cumulative GC statistics."""
    total_orphans: int
    batches_processed: int
    total_deleted: int
    total_freed_bytes: int
    total_errors: int
    elapsed_seconds: float


class IncrementalGC:
    """Incremental garbage collection manager."""
    
    def __init__(
        self,
        catalog: CatalogStore,
        root_dir: str,
        batch_size: int = 100,
    ):
        """Initialize incremental GC.
        
        Args:
            catalog: Catalog store
            root_dir: Root storage directory
            batch_size: Files to process per batch (default 100)
        """
        self.catalog = catalog
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self._orphans_cache: Optional[list[str]] = None
    
    def _find_all_orphans(self) -> list[str]:
        """Find all orphaned files (cached)."""
        if self._orphans_cache is not None:
            return self._orphans_cache
        
        logger.info("Scanning for orphaned files...")
        referenced = collect_referenced_paths(self.catalog)
        orphans = find_orphans(str(self.root_dir), referenced)
        
        logger.info(f"Found {len(orphans)} orphaned files")
        self._orphans_cache = orphans
        return orphans
    
    def gc_incremental(
        self,
        dry_run: bool = True,
        progress_callback: Optional[Callable[[GCBatchResult], None]] = None,
    ) -> GCStats:
        """Run incremental GC with batching.
        
        Args:
            dry_run: If True, don't actually delete files
            progress_callback: Optional callback(batch_result)
            
        Returns:
            GCStats with cumulative results
        """
        start_time = time.time()
        orphans = self._find_all_orphans()
        
        total_deleted = 0
        total_freed = 0
        total_errors = 0
        batch_num = 0
        
        # Process in batches
        for batch_start in range(0, len(orphans), self.batch_size):
            batch_num += 1
            batch_end = min(batch_start + self.batch_size, len(orphans))
            batch_files = orphans[batch_start:batch_end]
            
            batch_result = self._process_batch(
                batch_num, batch_files, dry_run
            )
            
            total_deleted += batch_result.files_deleted
            total_freed += batch_result.total_freed_bytes
            total_errors += batch_result.errors
            
            if progress_callback:
                progress_callback(batch_result)
            
            logger.info(
                f"Batch {batch_num}: deleted {batch_result.files_deleted} files "
                f"({batch_result.total_freed_bytes / 1024 / 1024:.1f}MB) "
                f"in {batch_result.elapsed_seconds:.1f}s"
            )
        
        elapsed = time.time() - start_time
        
        stats = GCStats(
            total_orphans=len(orphans),
            batches_processed=batch_num,
            total_deleted=total_deleted,
            total_freed_bytes=total_freed,
            total_errors=total_errors,
            elapsed_seconds=elapsed,
        )
        
        logger.info(
            f"GC complete: {total_deleted}/{len(orphans)} files deleted "
            f"({total_freed / 1024 / 1024 / 1024:.1f}GB freed) "
            f"in {elapsed:.1f}s"
        )
        
        return stats
    
    def _process_batch(
        self,
        batch_num: int,
        files: list[str],
        dry_run: bool,
    ) -> GCBatchResult:
        """Process a single batch of files."""
        batch_start = time.time()
        deleted = 0
        errors = 0
        freed_bytes = 0
        
        for file_path in files:
            try:
                path = Path(file_path)
                
                if not dry_run:
                    if path.exists():
                        freed_bytes += path.stat().st_size
                        path.unlink()
                    deleted += 1
                else:
                    # Dry-run: just count what would be deleted
                    if path.exists():
                        freed_bytes += path.stat().st_size
                    deleted += 1
            
            except Exception as e:
                logger.warning(f"Error deleting {file_path}: {e}")
                errors += 1
        
        elapsed = time.time() - batch_start
        
        return GCBatchResult(
            batch_num=batch_num,
            files_processed=len(files),
            files_deleted=deleted,
            errors=errors,
            elapsed_seconds=elapsed,
            total_freed_bytes=freed_bytes,
        )
    
    def gc_resume(
        self,
        last_batch: int,
        dry_run: bool = True,
        progress_callback: Optional[Callable[[GCBatchResult], None]] = None,
    ) -> GCStats:
        """Resume GC from a specific batch.
        
        Useful if GC was interrupted and you want to continue where it left off.
        
        Args:
            last_batch: Last batch that was processed (resume from next)
            dry_run: If True, don't actually delete
            progress_callback: Optional callback
            
        Returns:
            GCStats for resumed batches
        """
        start_time = time.time()
        orphans = self._find_all_orphans()
        
        # Skip to resumption point
        skip_files = last_batch * self.batch_size
        remaining_orphans = orphans[skip_files:]
        
        logger.info(
            f"Resuming GC from batch {last_batch + 1}, "
            f"processing {len(remaining_orphans)} remaining files"
        )
        
        total_deleted = 0
        total_freed = 0
        total_errors = 0
        batch_num = last_batch
        
        for batch_start in range(0, len(remaining_orphans), self.batch_size):
            batch_num += 1
            batch_end = min(batch_start + self.batch_size, len(remaining_orphans))
            batch_files = remaining_orphans[batch_start:batch_end]
            
            batch_result = self._process_batch(batch_num, batch_files, dry_run)
            
            total_deleted += batch_result.files_deleted
            total_freed += batch_result.total_freed_bytes
            total_errors += batch_result.errors
            
            if progress_callback:
                progress_callback(batch_result)
            
            logger.info(f"Batch {batch_num}: {batch_result.files_deleted} deleted")
        
        elapsed = time.time() - start_time
        
        return GCStats(
            total_orphans=len(remaining_orphans),
            batches_processed=batch_num - last_batch,
            total_deleted=total_deleted,
            total_freed_bytes=total_freed,
            total_errors=total_errors,
            elapsed_seconds=elapsed,
        )


def gc_incremental_cli(
    catalog: CatalogStore,
    root_dir: str,
    batch_size: int = 100,
    dry_run: bool = True,
) -> dict:
    """CLI-friendly incremental GC."""
    gc = IncrementalGC(catalog, root_dir, batch_size)
    
    # Progress callback for CLI
    def on_batch(batch: GCBatchResult):
        print(
            f"  Batch {batch.batch_num}: "
            f"{batch.files_deleted}/{batch.files_processed} deleted "
            f"({batch.total_freed_bytes / 1024 / 1024:.1f}MB)"
        )
    
    stats = gc.gc_incremental(dry_run=dry_run, progress_callback=on_batch)
    
    return {
        "total_orphans": stats.total_orphans,
        "batches_processed": stats.batches_processed,
        "total_deleted": stats.total_deleted,
        "total_freed_gb": stats.total_freed_bytes / 1024 / 1024 / 1024,
        "total_errors": stats.total_errors,
        "elapsed_seconds": stats.elapsed_seconds,
        "dry_run": dry_run,
    }
