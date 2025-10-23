# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.bootstrap",
#   "purpose": "Bootstrap and initialization for catalog system.",
#   "sections": [
#     {
#       "id": "build-catalog-store",
#       "name": "build_catalog_store",
#       "anchor": "function-build-catalog-store",
#       "kind": "function"
#     },
#     {
#       "id": "build-storage-layout",
#       "name": "build_storage_layout",
#       "anchor": "function-build-storage-layout",
#       "kind": "function"
#     },
#     {
#       "id": "get-catalog-root-dir",
#       "name": "get_catalog_root_dir",
#       "anchor": "function-get-catalog-root-dir",
#       "kind": "function"
#     },
#     {
#       "id": "catalogbootstrap",
#       "name": "CatalogBootstrap",
#       "anchor": "class-catalogbootstrap",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Bootstrap and initialization for catalog system.

Provides factory functions to initialize the catalog, storage layouts, and
all integration points with the main download pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from DocsToKG.ContentDownload.catalog.s3_layout import S3Layout
from DocsToKG.ContentDownload.catalog.store import CatalogStore, SQLiteCatalog
from DocsToKG.ContentDownload.config.models import CatalogConfig, StorageConfig

logger = logging.getLogger(__name__)


def build_catalog_store(config: CatalogConfig) -> CatalogStore:
    """Build a catalog store from config.

    Args:
        config: CatalogConfig instance

    Returns:
        Initialized CatalogStore (SQLiteCatalog or future Postgres)

    Raises:
        ValueError: If config is invalid or store initialization fails
    """
    if config.backend == "sqlite":
        logger.info(f"Initializing SQLite catalog at {config.path}")
        try:
            store = SQLiteCatalog(path=config.path, wal_mode=config.wal_mode)
            logger.info("Catalog store initialized successfully")
            return store
        except Exception as e:
            logger.error(f"Failed to initialize SQLite catalog: {e}")
            raise ValueError(f"Failed to initialize catalog: {e}") from e

    elif config.backend == "postgres":
        raise NotImplementedError("Postgres backend not yet implemented")

    else:
        raise ValueError(f"Unknown catalog backend: {config.backend}")


def build_storage_layout(config: StorageConfig) -> Optional[S3Layout]:
    """Build S3 storage layout if configured.

    Args:
        config: StorageConfig instance

    Returns:
        S3Layout instance if backend='s3', None otherwise

    Raises:
        ValueError: If S3 config is invalid
    """
    if config.backend == "fs":
        logger.debug("Using filesystem storage (no S3 layout)")
        return None

    elif config.backend == "s3":
        if not config.s3_bucket:
            raise ValueError("S3 backend requires s3_bucket to be configured")

        logger.info(f"Initializing S3 layout for bucket {config.s3_bucket}")
        layout = S3Layout(
            bucket=config.s3_bucket,
            prefix=config.s3_prefix,
            storage_class=config.s3_storage_class,
        )
        return layout

    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")


def get_catalog_root_dir(config: StorageConfig) -> str:
    """Get the root directory for artifact storage.

    Args:
        config: StorageConfig instance

    Returns:
        Root directory path
    """
    return config.root_dir


class CatalogBootstrap:
    """Orchestrates catalog system initialization and cleanup."""

    def __init__(self, catalog_config: CatalogConfig, storage_config: StorageConfig):
        """Initialize bootstrap with configs.

        Args:
            catalog_config: CatalogConfig instance
            storage_config: StorageConfig instance
        """
        self.catalog_config = catalog_config
        self.storage_config = storage_config
        self._catalog: Optional[CatalogStore] = None
        self._s3_layout: Optional[S3Layout] = None

    def initialize(self) -> CatalogBootstrap:
        """Initialize all catalog components.

        Returns:
            Self for method chaining
        """
        self._catalog = build_catalog_store(self.catalog_config)
        self._s3_layout = build_storage_layout(self.storage_config)

        logger.info(
            f"Catalog bootstrap complete: "
            f"layout={self.storage_config.layout}, "
            f"backend={self.storage_config.backend}"
        )
        return self

    @property
    def catalog(self) -> CatalogStore:
        """Get initialized catalog store.

        Raises:
            RuntimeError: If not initialized
        """
        if self._catalog is None:
            raise RuntimeError("Catalog not initialized. Call initialize() first.")
        return self._catalog

    @property
    def s3_layout(self) -> Optional[S3Layout]:
        """Get S3 layout if configured."""
        return self._s3_layout

    @property
    def root_dir(self) -> str:
        """Get storage root directory."""
        return get_catalog_root_dir(self.storage_config)

    def close(self) -> None:
        """Cleanup and close catalog connection."""
        if self._catalog:
            self._catalog.close()
            logger.debug("Catalog connection closed")

    def __enter__(self) -> CatalogBootstrap:
        """Context manager entry."""
        return self.initialize()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
