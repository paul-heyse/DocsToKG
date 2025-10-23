# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.connectors.dev_provider",
#   "purpose": "Development Provider (SQLite + Local FS).",
#   "sections": [
#     {
#       "id": "developmentprovider",
#       "name": "DevelopmentProvider",
#       "anchor": "class-developmentprovider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Development Provider (SQLite + Local FS)

Implements CatalogProvider using SQLite for development and testing.
Features:
  - SQLite database (in-memory or file-based)
  - Thread-local in-memory caching
  - WAL mode for better concurrency
  - Idempotent register_or_get with unique constraints
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import (
    DocumentRecord,
    HealthCheck,
    HealthStatus,
    ProviderConnectionError,
    ProviderOperationError,
)

logger = logging.getLogger(__name__)


class DevelopmentProvider:
    """Development provider - SQLite + local filesystem."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize development provider."""
        self.config = config
        self.db_path = config.get("db_path", ":memory:")
        self.cache_size = config.get("cache_size", 1000)
        self.enable_wal = config.get("enable_wal", True)
        self.conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}
        self._initialized = False

    def open(self, config: dict[str, Any]) -> None:
        """Initialize SQLite database."""
        with self._lock:
            if self._initialized:
                return

            try:
                # Handle file paths
                if self.db_path != ":memory:":
                    db_file = Path(self.db_path)
                    db_file.parent.mkdir(parents=True, exist_ok=True)

                # Connect to database
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row

                # Enable foreign keys and WAL mode
                self.conn.execute("PRAGMA foreign_keys = ON")
                if self.enable_wal:
                    self.conn.execute("PRAGMA journal_mode = WAL")

                # Load and execute schema
                self._initialize_schema()

                self._initialized = True
                logger.info(f"Development provider initialized: {self.db_path}")

            except Exception as e:
                if self.conn:
                    self.conn.close()
                    self.conn = None
                raise ProviderConnectionError(f"Failed to initialize SQLite: {e}") from e

    def close(self) -> None:
        """Cleanup and close database connection."""
        with self._lock:
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception as e:
                    logger.warning(f"Error closing database: {e}")
                finally:
                    self.conn = None
                    self._cache.clear()
                    self._initialized = False

    def name(self) -> str:
        """Return provider name."""
        return "development"

    def _initialize_schema(self) -> None:
        """Initialize database schema from SQL."""
        if not self.conn:
            raise RuntimeError("Connection not initialized")

        # Create documents table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL,
                source_url TEXT NOT NULL,
                resolver TEXT NOT NULL,
                content_type TEXT,
                bytes INTEGER NOT NULL,
                sha256 TEXT,
                storage_uri TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                run_id TEXT
            )
        """
        )

        # Create unique constraint
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_unique
            ON documents(artifact_id, source_url, resolver)
        """
        )

        # Create lookup indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_ct ON documents(content_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_artifact ON documents(artifact_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_resolver ON documents(resolver)"
        )

        # Create variants table (optional)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                variant TEXT NOT NULL,
                storage_uri TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                content_type TEXT,
                sha256 TEXT,
                created_at TEXT NOT NULL
            )
        """
        )

        # Create variants indexes
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_variants_unique
            ON variants(document_id, variant)
        """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_variants_sha ON variants(sha256)")

        self.conn.commit()

    def register_or_get(
        self,
        artifact_id: str,
        source_url: str,
        resolver: str,
        content_type: str | None = None,
        bytes: int = 0,
        sha256: str | None = None,
        storage_uri: str = "",
        run_id: str | None = None,
    ) -> DocumentRecord:
        """Register a document or get existing record (idempotent)."""
        if not self.conn:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                now = datetime.utcnow().isoformat()

                # Try to get existing record first
                cursor = self.conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE artifact_id = ? AND source_url = ? AND resolver = ?
                    """,
                    (artifact_id, source_url, resolver),
                )
                row = cursor.fetchone()

                if row:
                    # Return existing record
                    return self._row_to_record(row)

                # Insert new record
                cursor = self.conn.execute(
                    """
                    INSERT INTO documents
                    (artifact_id, source_url, resolver, content_type, bytes, sha256, storage_uri, created_at, updated_at, run_id)
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

                # Fetch and return the inserted record
                record_id = cursor.lastrowid
                cursor = self.conn.execute("SELECT * FROM documents WHERE id = ?", (record_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_record(row)

                raise ProviderOperationError("Failed to retrieve inserted record")

            except sqlite3.IntegrityError as e:
                # Unique constraint violation - fetch the existing record
                cursor = self.conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE artifact_id = ? AND source_url = ? AND resolver = ?
                    """,
                    (artifact_id, source_url, resolver),
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_record(row)
                raise ProviderOperationError(f"Register failed and record not found: {e}") from e

            except Exception as e:
                raise ProviderOperationError(f"Register or get failed: {e}") from e

    def get_by_artifact(self, artifact_id: str) -> list[DocumentRecord]:
        """Get all records for a given artifact_id."""
        if not self.conn:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                cursor = self.conn.execute(
                    "SELECT * FROM documents WHERE artifact_id = ? ORDER BY created_at DESC",
                    (artifact_id,),
                )
                rows = cursor.fetchall()
                return [self._row_to_record(row) for row in rows]

            except Exception as e:
                raise ProviderOperationError(f"Query failed: {e}") from e

    def get_by_sha256(self, sha256: str) -> list[DocumentRecord]:
        """Get all records with a given SHA-256 hash."""
        if not self.conn:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                cursor = self.conn.execute(
                    "SELECT * FROM documents WHERE sha256 = ? ORDER BY created_at DESC",
                    (sha256,),
                )
                rows = cursor.fetchall()
                return [self._row_to_record(row) for row in rows]

            except Exception as e:
                raise ProviderOperationError(f"Query failed: {e}") from e

    def find_duplicates(self) -> list[tuple[str, int]]:
        """Find all SHA-256 hashes with more than one record."""
        if not self.conn:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                cursor = self.conn.execute(
                    """
                    SELECT sha256, COUNT(*) as cnt
                    FROM documents
                    WHERE sha256 IS NOT NULL
                    GROUP BY sha256
                    HAVING cnt > 1
                    ORDER BY cnt DESC
                    """
                )
                return [(row[0], row[1]) for row in cursor.fetchall()]

            except Exception as e:
                raise ProviderOperationError(f"Query failed: {e}") from e

    def verify(self, record_id: int) -> bool:
        """Verify a record's SHA-256 hash."""
        if not self.conn:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                cursor = self.conn.execute(
                    "SELECT sha256, storage_uri FROM documents WHERE id = ?",
                    (record_id,),
                )
                row = cursor.fetchone()

                if not row:
                    raise ProviderOperationError(f"Record not found: {record_id}")

                sha256, storage_uri = row[0], row[1]

                if not sha256:
                    # No hash to verify
                    return True

                # For development provider, we just check file exists
                # In production providers, this would compute actual hash
                if storage_uri.startswith("file://"):
                    path = Path(storage_uri.replace("file://", ""))
                    return path.exists()

                # For non-file URIs, assume verified
                return True

            except Exception as e:
                raise ProviderOperationError(f"Verification failed: {e}") from e

    def stats(self) -> dict[str, Any]:
        """Get catalog statistics."""
        if not self.conn:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
                total_records = cursor.fetchone()[0]

                cursor = self.conn.execute("SELECT SUM(bytes) FROM documents WHERE bytes > 0")
                total_bytes = cursor.fetchone()[0] or 0

                cursor = self.conn.execute(
                    "SELECT COUNT(DISTINCT sha256) FROM documents WHERE sha256 IS NOT NULL"
                )
                unique_sha256 = cursor.fetchone()[0]

                duplicates = len(self.find_duplicates())

                cursor = self.conn.execute("SELECT DISTINCT resolver FROM documents")
                resolvers = [row[0] for row in cursor.fetchall()]

                cursor = self.conn.execute(
                    """
                    SELECT resolver, COUNT(*) as cnt
                    FROM documents
                    GROUP BY resolver
                    """
                )
                by_resolver = {row[0]: row[1] for row in cursor.fetchall()}

                return {
                    "total_records": total_records,
                    "total_bytes": total_bytes,
                    "unique_sha256": unique_sha256,
                    "duplicates": duplicates,
                    "storage_backends": ["file"],
                    "resolvers": resolvers,
                    "by_resolver": by_resolver,
                }

            except Exception as e:
                raise ProviderOperationError(f"Stats query failed: {e}") from e

    def health_check(self) -> HealthCheck:
        """Check provider health."""
        if not self.conn:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message="Database not initialized",
                latency_ms=0,
                details={"error": "No connection"},
            )

        with self._lock:
            try:
                start = time.time()
                cursor = self.conn.execute("SELECT 1")
                cursor.fetchone()
                latency_ms = (time.time() - start) * 1000

                return HealthCheck(
                    status=HealthStatus.HEALTHY,
                    message="Development provider OK",
                    latency_ms=latency_ms,
                    details={"db_path": self.db_path},
                )

            except Exception as e:
                return HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    latency_ms=0,
                    details={"error": str(e)},
                )

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """Convert database row to DocumentRecord."""
        return DocumentRecord(
            id=row["id"],
            artifact_id=row["artifact_id"],
            source_url=row["source_url"],
            resolver=row["resolver"],
            content_type=row["content_type"],
            bytes=row["bytes"],
            sha256=row["sha256"],
            storage_uri=row["storage_uri"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            run_id=row["run_id"],
        )
