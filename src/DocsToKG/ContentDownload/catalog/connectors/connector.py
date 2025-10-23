# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.connectors.connector",
#   "purpose": "Unified Catalog Connector - Factory Pattern.",
#   "sections": [
#     {
#       "id": "catalogconnector",
#       "name": "CatalogConnector",
#       "anchor": "class-catalogconnector",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""
Unified Catalog Connector - Factory Pattern

Provides a single entry point (CatalogConnector) that transparently delegates
to the appropriate provider backend (Development, Enterprise, or Cloud).

This follows the PR-4 Embedding Providers pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import CatalogProvider, DocumentRecord, HealthCheck, ProviderConfigError

if TYPE_CHECKING:
    pass


class CatalogConnector:
    """
    Unified entry point for all catalog backends.

    Provides transparent abstraction over:
    - DevelopmentProvider (SQLite + local FS)
    - EnterpriseProvider (Postgres + local FS)
    - CloudProvider (RDS + S3)

    Users interact with this class exclusively - the underlying provider
    is invisible to them.

    Example:
        with CatalogConnector("development", {}) as cat:
            record = cat.register_or_get(...)

        with CatalogConnector("enterprise", {
            "connection_url": "postgresql://..."
        }) as cat:
            records = cat.get_by_artifact("test:001")
    """

    def __init__(self, provider_type: str, config: dict[str, Any]) -> None:
        """
        Initialize connector with specified provider.

        Args:
            provider_type: "development" | "enterprise" | "cloud"
            config: Provider-specific configuration dict

        Raises:
            ProviderConfigError: If provider_type is invalid
        """
        self.provider_type = provider_type
        self.config = config
        self._provider: CatalogProvider | None = None

    def _create_provider(self) -> CatalogProvider:
        """
        Factory method - creates appropriate provider instance.

        Returns:
            Initialized provider instance

        Raises:
            ProviderConfigError: If provider_type is unknown
        """
        if self.provider_type == "development":
            from .dev_provider import DevelopmentProvider

            return DevelopmentProvider(self.config)

        elif self.provider_type == "enterprise":
            from .enterprise_provider import EnterpriseProvider

            return EnterpriseProvider(self.config)

        elif self.provider_type == "cloud":
            from .cloud_provider import CloudProvider

            return CloudProvider(self.config)

        else:
            raise ProviderConfigError(
                f"Unknown provider type: {self.provider_type}. "
                "Expected 'development', 'enterprise', or 'cloud'."
            )

    def open(self) -> None:
        """Initialize the backend."""
        if self._provider is None:
            self._provider = self._create_provider()
            self._provider.open(self.config)

    def close(self) -> None:
        """Cleanup and close backend connections."""
        if self._provider is not None:
            self._provider.close()
            self._provider = None

    def __enter__(self) -> CatalogConnector:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # CRUD Operations - all delegate to provider

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
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.register_or_get(
            artifact_id=artifact_id,
            source_url=source_url,
            resolver=resolver,
            content_type=content_type,
            bytes=bytes,
            sha256=sha256,
            storage_uri=storage_uri,
            run_id=run_id,
        )

    def get_by_artifact(self, artifact_id: str) -> list[DocumentRecord]:
        """Get all records for a given artifact_id."""
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.get_by_artifact(artifact_id)

    def get_by_sha256(self, sha256: str) -> list[DocumentRecord]:
        """Get all records with a given SHA-256 hash."""
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.get_by_sha256(sha256)

    def find_duplicates(self) -> list[tuple[str, int]]:
        """Find all SHA-256 hashes with more than one record."""
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.find_duplicates()

    def verify(self, record_id: int) -> bool:
        """Verify a record's SHA-256 hash."""
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.verify(record_id)

    def stats(self) -> dict[str, Any]:
        """Get catalog statistics."""
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.stats()

    def health_check(self) -> HealthCheck:
        """Check provider health."""
        if self._provider is None:
            raise RuntimeError("Connector not opened. Use 'with' or call open().")

        return self._provider.health_check()
