"""
Development Provider (SQLite + Local FS)

Stub for Phase 2 implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import DocumentRecord, HealthCheck


class DevelopmentProvider:
    """Development provider - SQLite + local filesystem (Phase 2)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize development provider."""
        self.config = config

    def open(self, config: Dict[str, Any]) -> None:
        """Phase 2: Initialize SQLite."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def close(self) -> None:
        """Phase 2: Cleanup."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def name(self) -> str:
        """Return provider name."""
        return "development"

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
        """Phase 2: Register or get record."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def get_by_artifact(self, artifact_id: str) -> List[DocumentRecord]:
        """Phase 2: Get records by artifact ID."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def get_by_sha256(self, sha256: str) -> List[DocumentRecord]:
        """Phase 2: Get records by SHA-256."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def find_duplicates(self) -> List[Tuple[str, int]]:
        """Phase 2: Find duplicates."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def verify(self, record_id: int) -> bool:
        """Phase 2: Verify record."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def stats(self) -> Dict[str, Any]:
        """Phase 2: Get statistics."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")

    def health_check(self) -> HealthCheck:
        """Phase 2: Check health."""
        raise NotImplementedError("Phase 2: Development provider implementation pending")
