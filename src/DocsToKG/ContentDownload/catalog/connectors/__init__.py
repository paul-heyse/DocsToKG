"""
Catalog Connectors Package

Unified entry point for all catalog backends using the provider pattern.
Provides a single factory (CatalogConnector) that transparently delegates
to the appropriate provider backend (Development, Enterprise, or Cloud).

QUICK START:

    from DocsToKG.ContentDownload.catalog.connectors import CatalogConnector

    # Development: SQLite + local FS (zero config)
    with CatalogConnector("development", {}) as cat:
        record = cat.register_or_get(artifact_id="test:001", ...)

    # Enterprise: Postgres + local FS (file/env config)
    with CatalogConnector("enterprise", {
        "connection_url": "postgresql://user:pass@host/dbname",
        "pool_size": 20,
    }) as cat:
        records = cat.get_by_artifact("test:001")
        duplicates = cat.find_duplicates("sha256_hash")

    # Cloud: RDS + S3 (auto-detect AWS credentials)
    with CatalogConnector("cloud", {
        "connection_url": "postgresql://rds-endpoint/catalog",
        "s3_bucket": "my-artifacts",
        "s3_prefix": "docs/",
    }) as cat:
        verified = cat.verify()
        stats = cat.stats()

PROVIDERS:
  - DevelopmentProvider: SQLite + local FS (development/testing)
  - EnterpriseProvider: Postgres + local FS (production on-premises)
  - CloudProvider: RDS + S3 (cloud/AWS deployments)

PROTOCOL:
  All providers implement CatalogProvider, supporting:
    open(), close(), register_or_get(), get_by_artifact(), get_by_sha256(),
    find_duplicates(), verify(), stats(), health_check()

CONFIGURATION:
  Each provider accepts a config dict with backend-specific keys.
  Use CatalogConfig (see ../config/) for multi-source loading (YAML/JSON/env).
"""

from __future__ import annotations

from .base import (
    CatalogProvider,
    DocumentRecord,
    HealthCheck,
    HealthStatus,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    ProviderOperationError,
)
from .connector import CatalogConnector
from .errors import (
    CloudProviderError,
    DevelopmentProviderError,
    EnterpriseProviderError,
)

__version__ = "1.0.0"

__all__ = [
    # Factory
    "CatalogConnector",
    # Protocol
    "CatalogProvider",
    # Types
    "DocumentRecord",
    "HealthStatus",
    "HealthCheck",
    # Exceptions
    "ProviderError",
    "ProviderConnectionError",
    "ProviderOperationError",
    "ProviderConfigError",
    "DevelopmentProviderError",
    "EnterpriseProviderError",
    "CloudProviderError",
]
