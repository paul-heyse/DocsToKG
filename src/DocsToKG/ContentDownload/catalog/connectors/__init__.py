"""
Catalog Connectors Package

Unified entry point for all catalog backends using the provider pattern.
Provides a single factory (CatalogConnector) that transparently delegates
to the appropriate provider backend.

Providers:
  - DevelopmentProvider: SQLite + local FS (development/testing)
  - EnterpriseProvider: Postgres + local FS (production on-premises)
  - CloudProvider: RDS + S3 (cloud/AWS deployments)

Usage:
    from DocsToKG.ContentDownload.catalog.connectors import CatalogConnector

    with CatalogConnector("development", {}) as cat:
        record = cat.register_or_get(...)
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
