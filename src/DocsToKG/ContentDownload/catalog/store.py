"""SQLite-based implementation of the artifact catalog store."""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from DocsToKG.ContentDownload.catalog.models import DocumentRecord

logger = logging.getLogger(__name__)


class CatalogStore:
    """Protocol-like base class for artifact catalog stores.

    This class defines the interface for catalog backends. Implementations
    should provide thread-safe CRUD operations for artifact metadata.
    """

    def register_or_get(
        self,
        *,
        artifact_id: str,
        source_url: str,
        resolver: str,
        content_type: Optional[str],
        bytes: int,
        sha256: Optional[str],
        storage_uri: str,
        run_id: Optional[str],
    ) -> DocumentRecord:
        """Register or retrieve a document record (idempotent).

        Args:
            artifact_id: Artifact identifier
            source_url: Source URL
            resolver: Resolver name
            content_type: MIME type
            bytes: File size
            sha256: SHA-256 hash (optional)
            storage_uri: Storage location URI
            run_id: Optional run ID

        Returns:
            DocumentRecord (existing if found, new if registered)

        Raises:
            ValueError: If validation fails
        """
        raise NotImplementedError

    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        """Get all records for an artifact ID."""
        raise NotImplementedError

    def get_by_sha256(self, sha256: str) -> List[DocumentRecord]:
        """Get all records with a given SHA-256 hash."""
        raise NotImplementedError

    def get_by_run(self, run_id: str) -> List[DocumentRecord]:
        """Get all records from a specific run."""
        raise NotImplementedError

    def find_duplicates(self) -> List[Tuple[str, int]]:
        """Find (sha256, count) tuples where count > 1."""
        raise NotImplementedError

    def verify(self, record_id: int) -> bool:
        """Verify SHA-256 of a record against stored file.

        Returns True if file exists and hash matches stored hash.
        Returns False if file missing or hash mismatch.
        """
        raise NotImplementedError

    def stats(self) -> Dict[str, int]:
        """Return catalog statistics."""
        raise NotImplementedError

    def get_all_records(self) -> List[DocumentRecord]:
        """Get all records in the catalog."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the store connection."""
        raise NotImplementedError


class SQLiteCatalog(CatalogStore):
    """SQLite-based implementation of the artifact catalog.

    Provides thread-safe CRUD operations with idempotent insertion,
    efficient lookups via indexes, and optional WAL mode for concurrency.
    """

    def __init__(self, path: str, wal_mode: bool = True):
        """Initialize SQLite catalog store.

        Args:
            path: Path to SQLite database file
            wal_mode: If True, enable WAL mode for better concurrency

        Raises:
            sqlite3.Error: If database initialization fails
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wal_mode = wal_mode
        self._lock = threading.RLock()

        # Initialize connection
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row

        if wal_mode:
            self.conn.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys=ON")

        # Load and execute schema
        self._init_schema()
        logger.info(f"Initialized SQLite catalog at {self.path}")

    def _init_schema(self) -> None:
        """Load and execute schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        schema_sql = schema_path.read_text()
        self.conn.executescript(schema_sql)
        self.conn.commit()
        logger.debug("Schema initialized successfully")

    def register_or_get(
        self,
        *,
        artifact_id: str,
        source_url: str,
        resolver: str,
        content_type: Optional[str],
        bytes: int,
        sha256: Optional[str],
        storage_uri: str,
        run_id: Optional[str],
    ) -> DocumentRecord:
        """Register or retrieve a document record (idempotent).

        Implementation uses INSERT OR IGNORE to handle idempotence.
        If the record already exists (same artifact_id, source_url, resolver),
        it is returned without modification.
        """
        with self._lock:
            now = datetime.utcnow().isoformat(timespec="seconds")

            try:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO documents
                    (artifact_id, source_url, resolver, content_type, bytes, sha256,
                     storage_uri, created_at, updated_at, run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        source_url,
                        resolver,
                        content_type,
                        bytes,
                        sha256,
                        storage_uri,
                        now,
                        now,
                        run_id,
                    ),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Failed to register document: {e}")
                raise ValueError(f"Failed to register document: {e}") from e

            # Retrieve the record
            cursor = self.conn.execute(
                """
                SELECT id, artifact_id, source_url, resolver, content_type, bytes,
                       sha256, storage_uri, created_at, updated_at, run_id
                FROM documents
                WHERE artifact_id = ? AND source_url = ? AND resolver = ?
                """,
                (artifact_id, source_url, resolver),
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError("Failed to retrieve registered document")

            return self._row_to_record(row)

    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        """Get all records for an artifact ID."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, artifact_id, source_url, resolver, content_type, bytes,
                       sha256, storage_uri, created_at, updated_at, run_id
                FROM documents
                WHERE artifact_id = ?
                ORDER BY created_at DESC
                """,
                (artifact_id,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_by_sha256(self, sha256: str) -> List[DocumentRecord]:
        """Get all records with a given SHA-256 hash."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, artifact_id, source_url, resolver, content_type, bytes,
                       sha256, storage_uri, created_at, updated_at, run_id
                FROM documents
                WHERE sha256 = ?
                ORDER BY created_at DESC
                """,
                (sha256,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_by_run(self, run_id: str) -> List[DocumentRecord]:
        """Get all records from a specific run."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, artifact_id, source_url, resolver, content_type, bytes,
                       sha256, storage_uri, created_at, updated_at, run_id
                FROM documents
                WHERE run_id = ?
                ORDER BY created_at DESC
                """,
                (run_id,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def find_duplicates(self) -> List[Tuple[str, int]]:
        """Find (sha256, count) tuples where count > 1."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT sha256, COUNT(*) as count
                FROM documents
                WHERE sha256 IS NOT NULL
                GROUP BY sha256
                HAVING count > 1
                ORDER BY count DESC
                """
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

    def verify(self, record_id: int) -> bool:
        """Verify SHA-256 of a record against stored file.

        Returns True if file exists and hash matches stored hash.
        Returns False if file missing or hash mismatch.
        """
        raise NotImplementedError

    def stats(self) -> Dict[str, int]:
        """Return catalog statistics."""
        with self._lock:
            cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            cursor = self.conn.execute(
                "SELECT COUNT(DISTINCT sha256) FROM documents WHERE sha256 IS NOT NULL"
            )
            unique_hashes = cursor.fetchone()[0]

            cursor = self.conn.execute("SELECT COUNT(DISTINCT artifact_id) FROM documents")
            unique_artifacts = cursor.fetchone()[0]

            cursor = self.conn.execute("SELECT SUM(bytes) FROM documents WHERE bytes > 0")
            total_bytes = cursor.fetchone()[0] or 0

            return {
                "total_documents": total_docs,
                "unique_hashes": unique_hashes,
                "unique_artifacts": unique_artifacts,
                "total_bytes": total_bytes,
            }

    def get_all_records(self) -> List[DocumentRecord]:
        """Get all records in the catalog."""
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT id, artifact_id, source_url, resolver, content_type, bytes,
                       sha256, storage_uri, created_at, updated_at, run_id
                FROM documents
                ORDER BY created_at DESC
                """
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self.conn:
                self.conn.close()
                logger.debug("Database connection closed")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        """Convert a database row to a DocumentRecord."""
        return DocumentRecord(
            id=row["id"],
            artifact_id=row["artifact_id"],
            source_url=row["source_url"],
            resolver=row["resolver"],
            content_type=row["content_type"],
            bytes=row["bytes"],
            sha256=row["sha256"],
            storage_uri=row["storage_uri"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            run_id=row["run_id"],
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
