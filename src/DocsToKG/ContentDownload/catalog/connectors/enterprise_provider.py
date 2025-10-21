"""
Enterprise Provider (Postgres + Local FS)

Stub for Phase 3 implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import DocumentRecord, HealthCheck, HealthStatus, ProviderConfigError


class EnterpriseProvider:
    """Enterprise provider - Postgres + local filesystem (Phase 3)."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enterprise provider."""
        self.config = config
    
    def open(self, config: Dict[str, Any]) -> None:
        """Phase 3: Initialize Postgres connection pool."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def close(self) -> None:
        """Phase 3: Cleanup."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def name(self) -> str:
        """Return provider name."""
        return "enterprise"
    
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
        """Phase 3: Register or get record."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        """Phase 3: Get records by artifact ID."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def get_by_sha256(self, sha256: str) -> List[DocumentRecord]:
        """Phase 3: Get records by SHA-256."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def find_duplicates(self) -> List[Tuple[str, int]]:
        """Phase 3: Find duplicates."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def verify(self, record_id: int) -> bool:
        """Phase 3: Verify record."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def stats(self) -> Dict[str, Any]:
        """Phase 3: Get statistics."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
    
    def health_check(self) -> HealthCheck:
        """Phase 3: Check health."""
        raise NotImplementedError("Phase 3: Enterprise provider implementation pending")
