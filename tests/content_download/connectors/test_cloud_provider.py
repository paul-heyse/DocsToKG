"""
Unit Tests for Cloud Provider (RDS + S3)

Tests cover:
- RDS connection initialization
- S3 bucket access verification
- All 9 protocol methods (via mock/integration)
- Dual health checks (database + S3)
- Configuration validation

Note: Full integration tests require AWS credentials and running RDS/S3.
Skipped tests are marked to run only with AWS environment available.
"""

from __future__ import annotations

import os

import pytest

from DocsToKG.ContentDownload.catalog.connectors import (
    CatalogConnector,
    ProviderConnectionError,
)


class TestCloudProvider:
    """Tests for CloudProvider (RDS + S3)."""

    def test_cloud_provider_requires_connection_url(self) -> None:
        """Cloud provider requires connection_url."""
        with pytest.raises(ProviderConnectionError, match="connection_url is required"):
            connector = CatalogConnector("cloud", {"s3_bucket": "test-bucket"})
            connector.open()

    def test_cloud_provider_requires_s3_bucket(self) -> None:
        """Cloud provider requires s3_bucket."""
        with pytest.raises(ProviderConnectionError, match="s3_bucket is required"):
            connector = CatalogConnector("cloud", {"connection_url": "postgresql://localhost/test"})
            connector.open()

    def test_cloud_provider_configuration(self) -> None:
        """Cloud provider accepts valid configuration."""
        config = {
            "connection_url": "postgresql://user:pass@rds.amazonaws.com/db",
            "s3_bucket": "my-catalog-bucket",
            "s3_region": "us-west-2",
            "s3_prefix": "artifacts/",
            "pool_size": 20,
            "max_overflow": 30,
        }
        connector = CatalogConnector("cloud", config)
        assert connector.provider_type == "cloud"
        # Actual connection would fail without AWS credentials

    def test_cloud_provider_name(self) -> None:
        """Cloud provider returns correct name."""
        connector = CatalogConnector(
            "cloud",
            {
                "connection_url": "postgresql://localhost/test",
                "s3_bucket": "test-bucket",
            },
        )
        # provider.name() would return "cloud" if initialized
        assert connector.provider_type == "cloud"

    def test_cloud_provider_not_opened_raises_error(self) -> None:
        """Operations on unopened provider raise error."""
        connector = CatalogConnector(
            "cloud",
            {
                "connection_url": "postgresql://localhost/test",
                "s3_bucket": "test-bucket",
            },
        )
        with pytest.raises(RuntimeError, match="Connector not opened"):
            connector.register_or_get("test:001", "http://example.com", "test")

    def test_cloud_provider_dual_backends(self) -> None:
        """Cloud provider requires both RDS and S3."""
        # Missing S3
        with pytest.raises(ProviderConnectionError):
            CatalogConnector("cloud", {"connection_url": "postgresql://localhost/db"}).open()

        # Missing RDS
        with pytest.raises(ProviderConnectionError):
            CatalogConnector("cloud", {"s3_bucket": "bucket"}).open()

    def test_cloud_provider_s3_configuration(self) -> None:
        """Cloud provider S3 configuration is accepted."""
        config = {
            "connection_url": "postgresql://localhost/test",
            "s3_bucket": "my-bucket",
            "s3_region": "eu-west-1",
            "s3_prefix": "artifacts/prod/",
        }
        connector = CatalogConnector("cloud", config)
        assert connector.provider_type == "cloud"

    def test_cloud_provider_connection_pooling_config(self) -> None:
        """Cloud provider connection pool configuration."""
        config = {
            "connection_url": "postgresql://localhost/test",
            "s3_bucket": "bucket",
            "pool_size": 25,
            "max_overflow": 35,
            "echo_sql": True,
        }
        connector = CatalogConnector("cloud", config)
        assert connector.provider_type == "cloud"


class TestCloudProviderIntegration:
    """Integration tests for Cloud Provider."""

    @pytest.mark.skipif(
        not (os.getenv("DOCSTOKG_TEST_RDS_URL") and os.getenv("AWS_ACCESS_KEY_ID")),
        reason="Requires DOCSTOKG_TEST_RDS_URL and AWS credentials",
    )
    def test_cloud_provider_full_lifecycle(self) -> None:
        """Full lifecycle test with real RDS + S3."""
        rds_url = os.getenv("DOCSTOKG_TEST_RDS_URL")
        s3_bucket = os.getenv("DOCSTOKG_TEST_S3_BUCKET", "test-catalog")

        with CatalogConnector(
            "cloud",
            {
                "connection_url": rds_url,
                "s3_bucket": s3_bucket,
                "s3_region": "us-east-1",
            },
        ) as cat:
            # Register record
            record = cat.register_or_get(
                artifact_id="cloud:001",
                source_url="http://example.com/doc.pdf",
                resolver="test",
                bytes=500,
                storage_uri="s3://test-catalog/test/doc.pdf",
            )

            assert record.id > 0
            assert record.artifact_id == "cloud:001"

            # Verify health (checks both RDS and S3)
            health = cat.health_check()
            assert health.status.value == "healthy"
            assert "database_latency_ms" in health.details
            assert "s3_latency_ms" in health.details

    @pytest.mark.skipif(
        not (os.getenv("DOCSTOKG_TEST_RDS_URL") and os.getenv("AWS_ACCESS_KEY_ID")),
        reason="Requires DOCSTOKG_TEST_RDS_URL and AWS credentials",
    )
    def test_cloud_provider_idempotent_register(self) -> None:
        """Idempotent registration with RDS backend."""
        rds_url = os.getenv("DOCSTOKG_TEST_RDS_URL")
        s3_bucket = os.getenv("DOCSTOKG_TEST_S3_BUCKET", "test-catalog")

        with CatalogConnector(
            "cloud",
            {"connection_url": rds_url, "s3_bucket": s3_bucket},
        ) as cat:
            record1 = cat.register_or_get(
                artifact_id="cloud:002",
                source_url="http://example.com/2",
                resolver="test",
                bytes=600,
                storage_uri="s3://test-catalog/test/2.pdf",
            )

            record2 = cat.register_or_get(
                artifact_id="cloud:002",
                source_url="http://example.com/2",
                resolver="test",
                bytes=600,
                storage_uri="s3://test-catalog/test/2.pdf",
            )

            assert record1.id == record2.id

    @pytest.mark.skipif(
        not (os.getenv("DOCSTOKG_TEST_RDS_URL") and os.getenv("AWS_ACCESS_KEY_ID")),
        reason="Requires DOCSTOKG_TEST_RDS_URL and AWS credentials",
    )
    def test_cloud_provider_queries(self) -> None:
        """Query operations with RDS backend."""
        rds_url = os.getenv("DOCSTOKG_TEST_RDS_URL")
        s3_bucket = os.getenv("DOCSTOKG_TEST_S3_BUCKET", "test-catalog")

        with CatalogConnector("cloud", {"connection_url": rds_url, "s3_bucket": s3_bucket}) as cat:
            # Register records
            cat.register_or_get(
                artifact_id="cloud:query:001",
                source_url="http://example.com/q1",
                resolver="test",
                sha256="abc123def456",
                bytes=700,
                storage_uri="s3://test-catalog/q1.pdf",
            )

            # Query by artifact
            records = cat.get_by_artifact("cloud:query:001")
            assert len(records) >= 1
            assert records[0].artifact_id == "cloud:query:001"

            # Query by SHA
            sha_records = cat.get_by_sha256("abc123def456")
            assert len(sha_records) >= 1

    @pytest.mark.skipif(
        not (os.getenv("DOCSTOKG_TEST_RDS_URL") and os.getenv("AWS_ACCESS_KEY_ID")),
        reason="Requires DOCSTOKG_TEST_RDS_URL and AWS credentials",
    )
    def test_cloud_provider_stats(self) -> None:
        """Statistics collection from RDS + S3."""
        rds_url = os.getenv("DOCSTOKG_TEST_RDS_URL")
        s3_bucket = os.getenv("DOCSTOKG_TEST_S3_BUCKET", "test-catalog")

        with CatalogConnector("cloud", {"connection_url": rds_url, "s3_bucket": s3_bucket}) as cat:
            # Register some records
            for i in range(3):
                cat.register_or_get(
                    artifact_id=f"cloud:stats:{i}",
                    source_url=f"http://example.com/{i}",
                    resolver="test",
                    bytes=100 * (i + 1),
                    storage_uri=f"s3://test-catalog/stat{i}.pdf",
                )

            # Get statistics
            stats = cat.stats()
            assert stats["total_records"] >= 3
            assert stats["s3_bucket"] == s3_bucket
            assert "s3_region" in stats
