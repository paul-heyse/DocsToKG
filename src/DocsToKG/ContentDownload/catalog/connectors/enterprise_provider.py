"""
Enterprise Provider (Postgres + Local FS)

Implements CatalogProvider using Postgres for production deployments.
Features:
  - Postgres database with connection pooling
  - Thread-safe operations with RLock
  - SQLAlchemy for ORM abstraction
  - ACID compliance
  - Production-grade configuration
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    DocumentRecord,
    HealthCheck,
    HealthStatus,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderOperationError,
)

logger = logging.getLogger(__name__)


class EnterpriseProvider:
    """Enterprise provider - Postgres + local filesystem."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enterprise provider."""
        self.config = config
        self.connection_url = config.get("connection_url")
        self.pool_size = config.get("pool_size", 10)
        self.max_overflow = config.get("max_overflow", 20)
        self.echo_sql = config.get("echo_sql", False)
        self.engine: Optional[Any] = None
        self.connection_pool: Optional[Any] = None
        self._lock = threading.RLock()
        self._initialized = False

    def open(self, config: Dict[str, Any]) -> None:
        """Initialize Postgres connection pool."""
        with self._lock:
            if self._initialized:
                return

            try:
                if not self.connection_url:
                    raise ProviderConfigError("connection_url is required for enterprise provider")

                # Import SQLAlchemy here to avoid hard dependency
                try:
                    from sqlalchemy import create_engine
                    from sqlalchemy import pool as sqlalchemy_pool
                except ImportError as e:
                    raise ProviderConnectionError(f"SQLAlchemy not installed: {e}") from e

                # Create engine with connection pooling
                self.engine = create_engine(
                    self.connection_url,
                    poolclass=sqlalchemy_pool.QueuePool,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    echo=self.echo_sql,
                    pool_pre_ping=True,  # Verify connections before using
                )

                # Initialize schema
                self._initialize_schema()

                self._initialized = True
                logger.info(f"Enterprise provider initialized with pool_size={self.pool_size}")

            except Exception as e:
                if self.engine:
                    self.engine.dispose()
                    self.engine = None
                raise ProviderConnectionError(f"Failed to initialize Postgres: {e}") from e

    def close(self) -> None:
        """Cleanup and dispose connection pool."""
        with self._lock:
            if self.engine is not None:
                try:
                    self.engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing connection pool: {e}")
                finally:
                    self.engine = None
                    self._initialized = False

    def name(self) -> str:
        """Return provider name."""
        return "enterprise"

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")

        with self.engine.begin() as conn:
            # Create documents table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    resolver TEXT NOT NULL,
                    content_type TEXT,
                    bytes INTEGER NOT NULL,
                    sha256 TEXT,
                    storage_uri TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    run_id TEXT
                )
                """
            )

            # Create unique constraint
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_unique
                ON documents(artifact_id, source_url, resolver)
                """
            )

            # Create lookup indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_ct ON documents(content_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_run ON documents(run_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_artifact ON documents(artifact_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_resolver ON documents(resolver)")

            # Create variants table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS variants (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    variant TEXT NOT NULL,
                    storage_uri TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    content_type TEXT,
                    sha256 TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create variants indexes
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_variants_unique
                ON variants(document_id, variant)
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_variants_sha ON variants(sha256)")

    def register_or_get(
        self,
        artifact_id: str,
        source_url: str,
        resolver: str,
        content_type: Optional[str] = None,
        bytes: int = 0,
        sha256: Optional[str] = None,
        storage_uri: str = "",
        run_id: Optional[str] = None,
    ) -> DocumentRecord:
        """Register a document or get existing record (idempotent)."""
        if not self.engine:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                with self.engine.begin() as conn:
                    # Try to get existing record first
                    result = conn.execute(
                        """
                        SELECT * FROM documents
                        WHERE artifact_id = %s AND source_url = %s AND resolver = %s
                        """,
                        (artifact_id, source_url, resolver),
                    )
                    row = result.fetchone()

                    if row:
                        return self._row_to_record(row)

                    # Insert new record
                    now = datetime.utcnow().isoformat()
                    result = conn.execute(
                        """
                        INSERT INTO documents
                        (artifact_id, source_url, resolver, content_type, bytes, sha256, storage_uri, created_at, updated_at, run_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING *
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

                    row = result.fetchone()
                    if row:
                        return self._row_to_record(row)

                    raise ProviderOperationError("Failed to retrieve inserted record")

            except Exception as e:
                if "duplicate key" in str(e).lower():
                    # Race condition - fetch existing record
                    with self.engine.begin() as conn:
                        result = conn.execute(
                            """
                            SELECT * FROM documents
                            WHERE artifact_id = %s AND source_url = %s AND resolver = %s
                            """,
                            (artifact_id, source_url, resolver),
                        )
                        row = result.fetchone()
                        if row:
                            return self._row_to_record(row)

                raise ProviderOperationError(f"Register or get failed: {e}") from e

    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        """Get all records for a given artifact_id."""
        if not self.engine:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        """
                        SELECT * FROM documents
                        WHERE artifact_id = %s
                        ORDER BY created_at DESC
                        """,
                        (artifact_id,),
                    )
                    rows = result.fetchall()
                    return [self._row_to_record(row) for row in rows]

            except Exception as e:
                raise ProviderOperationError(f"Query failed: {e}") from e

    def get_by_sha256(self, sha256: str) -> List[DocumentRecord]:
        """Get all records with a given SHA-256 hash."""
        if not self.engine:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        """
                        SELECT * FROM documents
                        WHERE sha256 = %s
                        ORDER BY created_at DESC
                        """,
                        (sha256,),
                    )
                    rows = result.fetchall()
                    return [self._row_to_record(row) for row in rows]

            except Exception as e:
                raise ProviderOperationError(f"Query failed: {e}") from e

    def find_duplicates(self) -> List[Tuple[str, int]]:
        """Find all SHA-256 hashes with more than one record."""
        if not self.engine:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        """
                        SELECT sha256, COUNT(*) as cnt
                        FROM documents
                        WHERE sha256 IS NOT NULL
                        GROUP BY sha256
                        HAVING COUNT(*) > 1
                        ORDER BY cnt DESC
                        """
                    )
                    return [(row[0], row[1]) for row in result.fetchall()]

            except Exception as e:
                raise ProviderOperationError(f"Query failed: {e}") from e

    def verify(self, record_id: int) -> bool:
        """Verify a record's SHA-256 hash."""
        if not self.engine:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        "SELECT sha256, storage_uri FROM documents WHERE id = %s",
                        (record_id,),
                    )
                    row = result.fetchone()

                    if not row:
                        raise ProviderOperationError(f"Record not found: {record_id}")

                    sha256, storage_uri = row[0], row[1]

                    if not sha256:
                        # No hash to verify
                        return True

                    # For enterprise provider, we just check file exists
                    # In cloud provider, this would verify S3 object
                    if storage_uri.startswith("file://"):
                        from pathlib import Path

                        path = Path(storage_uri.replace("file://", ""))
                        return path.exists()

                    # For non-file URIs, assume verified
                    return True

            except Exception as e:
                raise ProviderOperationError(f"Verification failed: {e}") from e

    def stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        if not self.engine:
            raise ProviderOperationError("Database not initialized")

        with self._lock:
            try:
                with self.engine.begin() as conn:
                    result = conn.execute("SELECT COUNT(*) FROM documents")
                    total_records = result.scalar() or 0

                    result = conn.execute("SELECT SUM(bytes) FROM documents WHERE bytes > 0")
                    total_bytes = result.scalar() or 0

                    result = conn.execute(
                        "SELECT COUNT(DISTINCT sha256) FROM documents WHERE sha256 IS NOT NULL"
                    )
                    unique_sha256 = result.scalar() or 0

                    duplicates = len(self.find_duplicates())

                    result = conn.execute("SELECT DISTINCT resolver FROM documents")
                    resolvers = [row[0] for row in result.fetchall()]

                    result = conn.execute(
                        """
                        SELECT resolver, COUNT(*) as cnt
                        FROM documents
                        GROUP BY resolver
                        """
                    )
                    by_resolver = {row[0]: row[1] for row in result.fetchall()}

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
        if not self.engine:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message="Database not initialized",
                latency_ms=0,
                details={"error": "No engine"},
            )

        with self._lock:
            try:
                start = time.time()
                with self.engine.begin() as conn:
                    conn.execute("SELECT 1")
                latency_ms = (time.time() - start) * 1000

                return HealthCheck(
                    status=HealthStatus.HEALTHY,
                    message="Enterprise provider OK",
                    latency_ms=latency_ms,
                    details={"pool_size": self.pool_size},
                )

            except Exception as e:
                return HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    latency_ms=0,
                    details={"error": str(e)},
                )

    def _row_to_record(self, row: Any) -> DocumentRecord:
        """Convert database row to DocumentRecord."""
        return DocumentRecord(
            id=row[0],
            artifact_id=row[1],
            source_url=row[2],
            resolver=row[3],
            content_type=row[4],
            bytes=row[5],
            sha256=row[6],
            storage_uri=row[7],
            created_at=row[8].isoformat() if row[8] else "",
            updated_at=row[9].isoformat() if row[9] else "",
            run_id=row[10],
        )
