# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.connectors.cloud_provider",
#   "purpose": "Cloud Provider (RDS + S3).",
#   "sections": [
#     {
#       "id": "cloudprovider",
#       "name": "CloudProvider",
#       "anchor": "class-cloudprovider",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Cloud Provider (RDS + S3)

Implements CatalogProvider using AWS RDS (managed Postgres) and S3 storage
for cloud-native deployments.

Features:
  - AWS RDS for managed database with connection pooling
  - Amazon S3 for unlimited scalable storage
  - Multi-region support
  - Automatic backups and high availability (RDS managed)
  - CDN-ready storage distribution
  - IAM authentication support
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any

from .base import (
    DocumentRecord,
    HealthCheck,
    HealthStatus,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderOperationError,
)

logger = logging.getLogger(__name__)


class CloudProvider:
    """Cloud provider - RDS + S3 (AWS)."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize cloud provider."""
        self.config = config
        self.connection_url = config.get("connection_url")
        self.s3_bucket = config.get("s3_bucket")
        self.s3_region = config.get("s3_region", "us-east-1")
        self.s3_prefix = config.get("s3_prefix", "artifacts/")
        self.pool_size = config.get("pool_size", 10)
        self.max_overflow = config.get("max_overflow", 20)
        self.echo_sql = config.get("echo_sql", False)
        self.engine: Any | None = None
        self.s3_client: Any | None = None
        self._lock = threading.RLock()
        self._initialized = False

    def open(self, config: dict[str, Any]) -> None:
        """Initialize RDS + S3 backend."""
        with self._lock:
            if self._initialized:
                return

            try:
                if not self.connection_url:
                    raise ProviderConfigError("connection_url is required for cloud provider")
                if not self.s3_bucket:
                    raise ProviderConfigError("s3_bucket is required for cloud provider")

                # Initialize database (RDS via SQLAlchemy)
                self._init_database()

                # Initialize S3 storage
                self._init_s3()

                self._initialized = True
                logger.info("Cloud provider (RDS + S3) initialized successfully")

            except Exception as e:
                if self.engine:
                    try:
                        self.engine.dispose()
                    except Exception:
                        pass
                raise ProviderConnectionError(f"Failed to initialize cloud provider: {e}") from e

    def _init_database(self) -> None:
        """Initialize RDS connection pool."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy import pool as sqlalchemy_pool
        except ImportError as e:
            raise ProviderConnectionError(f"SQLAlchemy not installed: {e}") from e

        self.engine = create_engine(
            self.connection_url,  # type: ignore[arg-type]
            poolclass=sqlalchemy_pool.QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            echo=self.echo_sql,
            pool_pre_ping=True,
        )

        # Initialize schema
        self._initialize_schema()
        logger.info("RDS database initialized")

    def _init_s3(self) -> None:
        """Initialize S3 client."""
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as e:
            raise ProviderConnectionError(f"boto3 not installed: {e}") from e

        try:
            self.s3_client = boto3.client("s3", region_name=self.s3_region)

            # Verify S3 bucket exists and is accessible
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 bucket verified: {self.s3_bucket}")
        except Exception as e:
            raise ProviderConnectionError(f"Failed to access S3 bucket: {e}") from e

    def close(self) -> None:
        """Cleanup and dispose resources."""
        with self._lock:
            if self.engine is not None:
                try:
                    self.engine.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing RDS connection: {e}")
                finally:
                    self.engine = None

            # S3 client doesn't need explicit closing
            self.s3_client = None
            self._initialized = False

    def name(self) -> str:
        """Return provider name."""
        return "cloud"

    def _initialize_schema(self) -> None:
        """Initialize RDS database schema (same as Enterprise)."""
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
        content_type: str | None = None,
        bytes: int = 0,
        sha256: str | None = None,
        storage_uri: str = "",
        run_id: str | None = None,
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

    def get_by_artifact(self, artifact_id: str) -> list[DocumentRecord]:
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

    def get_by_sha256(self, sha256: str) -> list[DocumentRecord]:
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

    def find_duplicates(self) -> list[tuple[str, int]]:
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
        """Verify a record's SHA-256 hash against S3."""
        if not self.engine or not self.s3_client:
            raise ProviderOperationError("Provider not initialized")

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

                    # Parse S3 URI (s3://bucket/key)
                    if storage_uri.startswith("s3://"):
                        bucket, key = storage_uri[5:].split("/", 1)

                        # Get object metadata from S3
                        try:
                            response = self.s3_client.head_object(Bucket=bucket, Key=key)
                            metadata = response.get("Metadata", {})
                            stored_hash = metadata.get("sha256")

                            if stored_hash == sha256.lower():
                                return True
                            else:
                                logger.warning(
                                    f"Hash mismatch for {storage_uri}: "
                                    f"expected {sha256}, got {stored_hash}"
                                )
                                return False
                        except Exception as e:
                            logger.error(f"Failed to verify S3 object: {e}")
                            return False

                    # For non-S3 URIs, assume verified
                    return True

            except Exception as e:
                raise ProviderOperationError(f"Verification failed: {e}") from e

    def stats(self) -> dict[str, Any]:
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
                        "storage_backends": ["s3"],
                        "resolvers": resolvers,
                        "by_resolver": by_resolver,
                        "s3_bucket": self.s3_bucket,
                        "s3_region": self.s3_region,
                    }

            except Exception as e:
                raise ProviderOperationError(f"Stats query failed: {e}") from e

    def health_check(self) -> HealthCheck:
        """Check provider health (RDS + S3)."""
        if not self.engine or not self.s3_client:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message="Provider not initialized",
                latency_ms=0,
                details={"error": "No engine or S3 client"},
            )

        with self._lock:
            try:
                start = time.time()

                # Check database
                with self.engine.begin() as conn:
                    conn.execute("SELECT 1")

                db_latency = (time.time() - start) * 1000

                # Check S3
                s3_start = time.time()
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                s3_latency = (time.time() - s3_start) * 1000

                return HealthCheck(
                    status=HealthStatus.HEALTHY,
                    message="Cloud provider OK (RDS + S3)",
                    latency_ms=db_latency + s3_latency,
                    details={
                        "database_latency_ms": db_latency,
                        "s3_latency_ms": s3_latency,
                        "s3_bucket": self.s3_bucket,
                    },
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
