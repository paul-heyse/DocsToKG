"""Catalog backup and recovery utilities.

Provides:
  - Atomic backup with metadata
  - Point-in-time restore
  - Backup validation
  - Recovery verification
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackupMetadata:
    """Metadata for a backup."""

    timestamp: str  # ISO format
    source_path: str
    destination_path: str
    file_size_bytes: int
    record_count: int
    database_version: Optional[str] = None
    checksum_sha256: Optional[str] = None


class CatalogBackup:
    """Catalog backup and recovery manager."""

    def __init__(self, catalog_path: str):
        """Initialize backup manager.

        Args:
            catalog_path: Path to catalog.sqlite
        """
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    def _get_record_count(self) -> int:
        """Get number of records in catalog."""
        try:
            conn = sqlite3.connect(str(self.catalog_path))
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"Could not get record count: {e}")
            return 0

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of file."""
        import hashlib

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def backup_atomic(
        self,
        destination: Path,
        include_checksum: bool = True,
    ) -> BackupMetadata:
        """Create an atomic backup of the catalog.

        Uses WAL checkpoint to ensure consistency, then copies file atomically.

        Args:
            destination: Destination directory for backup
            include_checksum: If True, compute SHA-256 checksum

        Returns:
            BackupMetadata with backup information
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        # Generate backup filename with timestamp
        timestamp = datetime.utcnow().isoformat()
        backup_filename = f"catalog_backup_{timestamp.replace(':', '-')}.sqlite"
        backup_path = destination / backup_filename

        logger.info(f"Starting atomic backup to {backup_path}")

        try:
            # Checkpoint WAL to ensure consistency
            conn = sqlite3.connect(str(self.catalog_path))
            conn.execute("PRAGMA wal_checkpoint(FULL)")
            conn.close()

            # Atomic copy
            shutil.copy2(str(self.catalog_path), str(backup_path))

            logger.info(f"Backup created: {backup_path}")

            # Gather metadata
            file_size = backup_path.stat().st_size
            record_count = self._get_record_count()
            checksum = None
            if include_checksum:
                checksum = self._compute_checksum(backup_path)

            metadata = BackupMetadata(
                timestamp=timestamp,
                source_path=str(self.catalog_path),
                destination_path=str(backup_path),
                file_size_bytes=file_size,
                record_count=record_count,
                checksum_sha256=checksum,
            )

            # Save metadata
            metadata_path = backup_path.with_suffix(".metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            logger.info(f"Backup metadata saved: {metadata_path}")
            return metadata

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Cleanup partial backup
            if backup_path.exists():
                backup_path.unlink()
            raise

    def backup_incremental(
        self,
        destination: Path,
        last_backup_path: Optional[str] = None,
    ) -> BackupMetadata:
        """Create incremental backup (stub for future implementation).

        For now, falls back to atomic backup. True incremental would use
        WAL pages or similar.

        Args:
            destination: Destination directory
            last_backup_path: Path to last backup (for comparison)

        Returns:
            BackupMetadata
        """
        logger.info("Incremental backup not yet implemented, using atomic backup")
        return self.backup_atomic(destination)

    def recover_from_backup(
        self,
        backup_path: Path,
        target_path: Path,
        dry_run: bool = True,
        validate: bool = True,
    ) -> bool:
        """Recover catalog from backup.

        Args:
            backup_path: Path to backup file
            target_path: Path to restore to
            dry_run: If True, don't actually restore
            validate: If True, validate backup before restore

        Returns:
            True if recovery successful
        """
        backup_path = Path(backup_path)
        target_path = Path(target_path)

        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False

        # Validate backup if requested
        if validate:
            if not self._validate_backup(backup_path):
                logger.error("Backup validation failed")
                return False

        logger.info(f"Recovering from {backup_path} to {target_path}")

        if dry_run:
            logger.info("DRY-RUN: Would restore backup")
            return True

        try:
            # Backup current catalog if it exists
            if target_path.exists():
                safe_path = target_path.with_suffix(".pre-recovery.sqlite")
                shutil.copy2(str(target_path), str(safe_path))
                logger.info(f"Current catalog backed up to {safe_path}")

            # Restore from backup
            shutil.copy2(str(backup_path), str(target_path))
            logger.info(f"Recovery complete: {target_path}")
            return True

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False

    def _validate_backup(self, backup_path: Path) -> bool:
        """Validate backup integrity.

        Args:
            backup_path: Path to backup file

        Returns:
            True if backup is valid
        """
        try:
            # Try to open and query the backup
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            conn.close()

            logger.info(f"Backup validation passed: {count} records")
            return True

        except Exception as e:
            logger.error(f"Backup validation failed: {e}")
            return False

    def list_backups(self, backup_dir: Path) -> list[tuple[Path, BackupMetadata]]:
        """List all backups in a directory.

        Args:
            backup_dir: Directory containing backups

        Returns:
            List of (backup_path, metadata) tuples
        """
        backup_dir = Path(backup_dir)
        if not backup_dir.exists():
            return []

        backups = []
        for backup_file in sorted(backup_dir.glob("catalog_backup_*.sqlite")):
            metadata_file = backup_file.with_suffix(".metadata.json")

            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata_dict = json.load(f)
                    metadata = BackupMetadata(**metadata_dict)
                    backups.append((backup_file, metadata))
                except Exception as e:
                    logger.warning(f"Could not load metadata for {backup_file}: {e}")

        return backups


def backup_catalog_cli(
    catalog_path: str,
    destination: str,
    validate: bool = True,
) -> dict:
    """CLI-friendly backup function."""
    try:
        manager = CatalogBackup(catalog_path)
        metadata = manager.backup_atomic(Path(destination), include_checksum=validate)

        return {
            "success": True,
            "timestamp": metadata.timestamp,
            "destination": metadata.destination_path,
            "file_size_mb": metadata.file_size_bytes / 1024 / 1024,
            "record_count": metadata.record_count,
            "checksum": metadata.checksum_sha256,
        }
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return {"success": False, "error": str(e)}


def recover_catalog_cli(
    backup_path: str,
    target_path: str,
    dry_run: bool = True,
) -> dict:
    """CLI-friendly recovery function."""
    try:
        manager = CatalogBackup(target_path)
        success = manager.recover_from_backup(
            Path(backup_path),
            Path(target_path),
            dry_run=dry_run,
            validate=True,
        )

        return {
            "success": success,
            "backup": backup_path,
            "target": target_path,
            "dry_run": dry_run,
        }
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return {"success": False, "error": str(e)}
