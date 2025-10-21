"""
Catalog Provider Protocol and Base Types

Defines the interface all catalog providers (Development, Enterprise, Cloud)
must implement. This follows the PR-4 Embedding Providers pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol
from enum import Enum


# Exception Types

class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass


class ProviderConnectionError(ProviderError):
    """Connection to backend failed."""
    pass


class ProviderOperationError(ProviderError):
    """Operation on backend failed."""
    pass


class ProviderConfigError(ProviderError):
    """Invalid provider configuration."""
    pass


# Health Status

class HealthStatus(Enum):
    """Health status of a catalog provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthCheck:
    """Health check result."""
    status: HealthStatus
    message: str
    latency_ms: float
    details: Dict[str, Any]


# Document Record (used by all providers)

@dataclass(frozen=True)
class DocumentRecord:
    """A catalog record for a stored document."""
    id: int
    artifact_id: str
    source_url: str
    resolver: str
    content_type: Optional[str]
    bytes: int
    sha256: Optional[str]
    storage_uri: str
    created_at: str
    updated_at: str
    run_id: Optional[str]


# Provider Protocol

class CatalogProvider(Protocol):
    """
    Protocol that all catalog providers must implement.
    
    Providers are responsible for:
    - Managing the catalog backend (SQLite, Postgres, RDS)
    - Handling storage operations (local FS, S3)
    - Thread safety and connection management
    - CRUD operations on catalog records
    """
    
    def name(self) -> str:
        """Return provider name: 'development' | 'enterprise' | 'cloud'."""
        ...
    
    def open(self, config: Dict[str, Any]) -> None:
        """
        Initialize the backend.
        
        Args:
            config: Provider-specific configuration dict
            
        Raises:
            ProviderConfigError: If config is invalid
            ProviderConnectionError: If backend connection fails
        """
        ...
    
    def close(self) -> None:
        """
        Cleanup and close backend connections.
        
        Safe to call multiple times.
        """
        ...
    
    def register_or_get(
        self,
        artifact_id: str,
        source_url: str,
        resolver: str,
        content_type: Optional[str],
        bytes: int,
        sha256: Optional[str],
        storage_uri: str,
        run_id: Optional[str],
    ) -> DocumentRecord:
        """
        Register a document or get existing record (idempotent).
        
        Uses (artifact_id, source_url, resolver) as unique key.
        
        Args:
            artifact_id: Unique artifact identifier (e.g., "doi:10.1234/...")
            source_url: URL where document was retrieved from
            resolver: Name of resolver that found this document
            content_type: MIME type of document
            bytes: File size in bytes
            sha256: SHA-256 hash (hex encoded) if computed
            storage_uri: Final storage location (file:/// or s3://)
            run_id: Run identifier for provenance
            
        Returns:
            DocumentRecord with database ID
            
        Raises:
            ProviderOperationError: If operation fails
        """
        ...
    
    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        """
        Get all records for a given artifact_id.
        
        Args:
            artifact_id: The artifact identifier
            
        Returns:
            List of records (may be empty)
            
        Raises:
            ProviderOperationError: If query fails
        """
        ...
    
    def get_by_sha256(self, sha256: str) -> List[DocumentRecord]:
        """
        Get all records with a given SHA-256 hash.
        
        Args:
            sha256: SHA-256 hash (hex encoded)
            
        Returns:
            List of records (may be empty)
            
        Raises:
            ProviderOperationError: If query fails
        """
        ...
    
    def find_duplicates(self) -> List[Tuple[str, int]]:
        """
        Find all SHA-256 hashes with more than one record.
        
        Returns:
            List of (sha256, count) tuples where count > 1
            
        Raises:
            ProviderOperationError: If query fails
        """
        ...
    
    def verify(self, record_id: int) -> bool:
        """
        Verify a record's SHA-256 hash.
        
        Args:
            record_id: The record ID to verify
            
        Returns:
            True if hash matches, False if mismatch
            
        Raises:
            ProviderOperationError: If verification fails
        """
        ...
    
    def stats(self) -> Dict[str, Any]:
        """
        Get catalog statistics.
        
        Returns dict with keys like:
        - total_records: int
        - total_bytes: int
        - unique_sha256: int
        - duplicates: int
        - storage_backends: [str]
        - by_resolver: {str: int}
        """
        ...
    
    def health_check(self) -> HealthCheck:
        """
        Check provider health.
        
        Returns:
            HealthCheck with status and details
        """
        ...
